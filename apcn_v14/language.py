from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Counter as CounterType, DefaultDict, Dict, List, Optional, Sequence, Tuple
import json
import math
import re

from apcn_v10.language_common import LanguageEpisode, tokenize
from apcn_v10.semantic import EntityRef, SemanticNode
from apcn_v11.language import ConstructionInducer
from apcn_v12.language import AdaptiveConstructionCalibrator, AdaptiveLanguageSessionV12, SemanticLanguageLearnerV12
from .language_teacher import RichSemanticTeacherV14

_SLOT_RE = re.compile(r"^<([ER])(\d+)>$")


class ProgramConstructionMemory:
    """Sparse bounded surface-construction -> semantic-program memory."""

    VERSION = "APCN-V0.14-PROGRAM-CONSTRUCTION-MEMORY"

    def __init__(self, max_patterns: int = 4096):
        self.max_patterns = int(max_patterns)
        self.pattern_schema: DefaultDict[str, CounterType[str]] = defaultdict(Counter)
        self.touches: CounterType[str] = Counter()
        self.observations = 0
        self.skipped_unaligned = 0

    @staticmethod
    def _relation_spans(learner, tokens: Sequence[str]) -> List[Tuple[int, int, str, float]]:
        candidates: List[Tuple[float, int, int, str]] = []
        for n in range(1, min(5, len(tokens)) + 1):
            for i in range(len(tokens)-n+1):
                cue = " ".join(tokens[i:i+n])
                feat, score = learner.best_feature(cue, ("relation:",))
                if feat is None or score < .10:
                    continue
                purity = learner.feature_purity(cue, feat)
                support = learner.cue_support(cue, feat)
                if purity < .46 or support < 3:
                    continue
                value = float(score * (1.0 + .08*(n-1)) * (.65 + .35*purity))
                candidates.append((value, i, i+n, feat.split(":",1)[1]))
        candidates.sort(reverse=True)
        chosen: List[Tuple[int,int,str,float]] = []
        occupied = set()
        for value, i, j, relation in candidates:
            span = set(range(i,j))
            if occupied & span:
                continue
            chosen.append((i,j,relation,value)); occupied |= span
        chosen.sort(key=lambda x:x[0])
        return chosen

    @staticmethod
    def _numbered_pattern(tokens: Sequence[str], entity_mentions, relation_spans) -> str:
        starts: Dict[int, Tuple[int,str]] = {}
        for idx,(i,j,_) in enumerate(entity_mentions):
            starts[i] = max(starts.get(i,(0,"")), (j-i,f"<E{idx}>"))
        for idx,(i,j,_,_) in enumerate(relation_spans):
            cand=(j-i,f"<R{idx}>")
            if i not in starts or cand[0] > starts[i][0]: starts[i]=cand
        out=[]; i=0
        while i < len(tokens):
            row=starts.get(i)
            if row is None: out.append(tokens[i]); i+=1; continue
            length,slot=row; out.append(slot); i+=length
        return " ".join(out)

    @staticmethod
    def _schema_for(node: SemanticNode, entity_map: Dict[Tuple[str,str],int], relation_map: Dict[str,int]):
        if node.op == "RELATION":
            if node.subject is None or node.object is None or node.relation is None: return None
            s=entity_map.get((node.subject.color,node.subject.shape)); o=entity_map.get((node.object.color,node.object.shape)); r=relation_map.get(node.relation)
            if s is None or o is None or r is None: return None
            return {"op":"RELATION","r":r,"s":s,"o":o}
        children=[]
        for child in node.children:
            row=ProgramConstructionMemory._schema_for(child,entity_map,relation_map)
            if row is None: return None
            children.append(row)
        return {"op":node.op,"children":children}

    def align(self, learner, utterance: str, program: Optional[SemanticNode]=None):
        tokens=tokenize(utterance); entities=learner._entity_mentions(tokens); relations=self._relation_spans(learner,tokens)
        if len(entities)<2 or not relations: return None
        desc=[(m[2].color,m[2].shape) for m in entities]
        if len(set(desc)) != len(desc): return None
        pattern=self._numbered_pattern(tokens,entities,relations)
        if program is None: return pattern,entities,relations,None
        entity_map={d:i for i,d in enumerate(desc)}; relation_map={}
        for i,(_,_,rel,_) in enumerate(relations): relation_map.setdefault(rel,i)
        schema=self._schema_for(program,entity_map,relation_map)
        if schema is None: return None
        return pattern,entities,relations,schema

    @staticmethod
    def _schema_key(schema: Dict[str,object]) -> str:
        return json.dumps(schema,sort_keys=True,separators=(",",":"))

    @staticmethod
    def _decision(counter: CounterType[str]):
        total=int(sum(counter.values()))
        if total<=0: return None,0.0,0
        ordered=counter.most_common(2); key,top=ordered[0]; second=ordered[1][1] if len(ordered)>1 else 0
        purity=top/total; margin=(top-second)/total; support=1.0-math.exp(-total/4.0)
        return key,float(purity*(.72+.28*margin)*support),total

    @staticmethod
    def _generic_tokens(pattern: str) -> List[str]:
        out=[]
        for tok in pattern.split():
            m=_SLOT_RE.match(tok); out.append(f"<{m.group(1)}>" if m else tok)
        return out

    @staticmethod
    def _slot_counts(pattern: str):
        e=r=0
        for tok in pattern.split():
            m=_SLOT_RE.match(tok)
            if not m: continue
            if m.group(1)=="E": e=max(e,int(m.group(2))+1)
            else: r=max(r,int(m.group(2))+1)
        return e,r

    @classmethod
    def _similarity(cls,a: str,b: str) -> float:
        if cls._slot_counts(a)!=cls._slot_counts(b): return 0.0
        aa,bb=cls._generic_tokens(a),cls._generic_tokens(b)
        seq=SequenceMatcher(None,aa,bb).ratio(); la={x for x in aa if not x.startswith("<")}; lb={x for x in bb if not x.startswith("<")}; union=la|lb
        jac=len(la&lb)/len(union) if union else 1.0
        return float(.74*seq+.26*jac)

    def observe(self, learner, episode: LanguageEpisode) -> bool:
        aligned=self.align(learner,episode.utterance,episode.program)
        if aligned is None: self.skipped_unaligned+=1; return False
        pattern,_,_,schema=aligned; key=self._schema_key(schema)
        self.pattern_schema[pattern][key]+=1; self.touches[pattern]+=1; self.observations+=1
        if len(self.pattern_schema)>self.max_patterns:
            ranked=sorted(self.pattern_schema,key=lambda p:(self.touches[p],sum(self.pattern_schema[p].values())))
            for p in ranked[:len(self.pattern_schema)-self.max_patterns]: self.pattern_schema.pop(p,None); self.touches.pop(p,None)
        return True

    def predict(self, learner, utterance: str):
        aligned=self.align(learner,utterance,None)
        if aligned is None: return None,0.0,"",None
        pattern,entities,relations,_=aligned; exact=self.pattern_schema.get(pattern)
        if exact:
            key,conf,support=self._decision(exact)
            if key is not None and support>=2: return json.loads(key),conf,pattern,(entities,relations)
        candidates=[]
        for known,counter in self.pattern_schema.items():
            key,conf,support=self._decision(counter)
            if key is None or support<3: continue
            sim=self._similarity(pattern,known)
            if sim<.58: continue
            candidates.append((conf*sim,sim,support,known,key))
        if not candidates: return None,0.0,pattern,(entities,relations)
        candidates.sort(reverse=True); conf,_,_,evidence,key=candidates[0]
        return json.loads(key),float(min(1.0,conf)),evidence,(entities,relations)

    @staticmethod
    def instantiate(schema: Dict[str,object],entities,relations) -> Optional[SemanticNode]:
        op=str(schema.get("op",""))
        if op=="RELATION":
            try: s=entities[int(schema["s"])][2]; o=entities[int(schema["o"])][2]; r=relations[int(schema["r"])][2]
            except (KeyError,IndexError,TypeError,ValueError): return None
            return SemanticNode("RELATION",relation=r,subject=s,object=o)
        children=[]
        for child in schema.get("children",[]):
            node=ProgramConstructionMemory.instantiate(child,entities,relations)
            if node is None: return None
            children.append(node)
        return SemanticNode(op,children=tuple(children)) if op else None

    def summary(self,limit: int=16) -> Dict[str,object]:
        rows=[]
        for pattern,counter in self.pattern_schema.items():
            key,conf,support=self._decision(counter)
            if key is not None: rows.append((conf,support,pattern,json.loads(key)))
        rows.sort(reverse=True)
        return {"version":self.VERSION,"observations":self.observations,"patterns":len(self.pattern_schema),"max_patterns":self.max_patterns,"skipped_unaligned":self.skipped_unaligned,
                "strongest":[{"pattern":p,"confidence":c,"support":s,"schema":schema} for c,s,p,schema in rows[:limit]]}

    def to_dict(self) -> Dict[str,object]:
        return {"version":self.VERSION,"max_patterns":self.max_patterns,"observations":self.observations,"skipped_unaligned":self.skipped_unaligned,
                "pattern_schema":{k:dict(v) for k,v in self.pattern_schema.items()},"touches":dict(self.touches)}

    @classmethod
    def from_dict(cls,data: Dict[str,object]) -> "ProgramConstructionMemory":
        obj=cls(int(data.get("max_patterns",4096))); obj.observations=int(data.get("observations",0)); obj.skipped_unaligned=int(data.get("skipped_unaligned",0))
        obj.pattern_schema=defaultdict(Counter,{k:Counter({x:int(y) for x,y in v.items()}) for k,v in data.get("pattern_schema",{}).items()}); obj.touches=Counter({k:int(v) for k,v in data.get("touches",{}).items()})
        return obj


class SemanticLanguageLearnerV14(SemanticLanguageLearnerV12):
    VERSION="APCN-V0.14-SEMANTIC-LANGUAGE-MEMORY"
    def __init__(self):
        super().__init__(); self.program_constructions=ProgramConstructionMemory(); self.last_program_evidence={}
    def observe(self,episode: LanguageEpisode) -> None:
        super().observe(episode); self.program_constructions.observe(self,episode)
    def parse(self,utterance: str,discourse_focus: Optional[EntityRef]=None,allow_sequence: bool=True,discourse_registry=None) -> Optional[SemanticNode]:
        schema,confidence,evidence,bindings=self.program_constructions.predict(self,utterance)
        self.last_program_evidence={"utterance":utterance,"confidence":confidence,"evidence":evidence,"used":False}
        if schema is not None and bindings is not None and confidence>=.57:
            node=self.program_constructions.instantiate(schema,bindings[0],bindings[1])
            if node is not None:
                if discourse_registry is not None: node=self._bind_discourse(node,tokenize(utterance),discourse_registry)
                self.last_program_evidence.update({"used":True,"schema":schema}); return node
        return super().parse(utterance,discourse_focus,allow_sequence,discourse_registry)
    @classmethod
    def from_v12(cls,old: SemanticLanguageLearnerV12) -> "SemanticLanguageLearnerV14":
        obj=cls(); obj.episode_count=int(old.episode_count); obj.feature_totals=Counter(old.feature_totals); obj.cue_totals=Counter(old.cue_totals)
        obj.cue_feature=defaultdict(Counter,{k:Counter(v) for k,v in old.cue_feature.items()}); obj.pos_feature=defaultdict(Counter,{k:Counter(v) for k,v in old.pos_feature.items()}); obj.pos_totals=Counter(old.pos_totals)
        obj.constructions=ConstructionInducer.from_dict(old.constructions.to_dict()); obj.adaptive_constructions=AdaptiveConstructionCalibrator.from_dict(old.adaptive_constructions.to_dict()); return obj
    def save(self,path: str|Path) -> None:
        super().save(path); p=Path(path); data=json.loads(p.read_text(encoding="utf-8")); data["version"]=self.VERSION; data["program_constructions"]=self.program_constructions.to_dict(); p.write_text(json.dumps(data,indent=2),encoding="utf-8")
    @classmethod
    def load(cls,path: str|Path) -> "SemanticLanguageLearnerV14":
        old=SemanticLanguageLearnerV12.load(path); obj=cls.from_v12(old); data=json.loads(Path(path).read_text(encoding="utf-8")); obj.program_constructions=ProgramConstructionMemory.from_dict(data.get("program_constructions",{})); return obj


class AdaptiveLanguageSessionV14(AdaptiveLanguageSessionV12):
    def __init__(self,seed: int=14,learner: Optional[SemanticLanguageLearnerV14]=None):
        super().__init__(seed=seed,learner=learner or SemanticLanguageLearnerV14()); self.teacher=RichSemanticTeacherV14(seed)
    def teach_user_paraphrase(self,utterance: str,program: SemanticNode,discourse_focus: Optional[EntityRef]=None):
        before=self.learner.parse(utterance,discourse_focus,discourse_registry=self.discourse); self.learner.observe(LanguageEpisode(utterance,program,"user_paraphrase",discourse_focus=discourse_focus)); return before

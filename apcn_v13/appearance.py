from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearanceLayout:
    dim: int
    groups: Dict[str, Tuple[int, int]]


class FineAppearanceEncoder:
    """Generic high-detail descriptor for persistent INSTANCE identity.

    V0.12 intentionally compresses perception for category/color/shape learning.
    V0.13 keeps that path and adds this separate, finer representation for
    distinguishing individual objects within one category.

    No instance labels, category labels or semantic color/shape bins enter this
    encoder. It uses only a focused image region and optional attention mask:
    local-contrast luminance, brightness-normalized chromaticity, edge fields and
    occupancy. The output is read-only and contains no learned weights.
    """

    VERSION = "APCN-V0.13-FINE-INSTANCE-APPEARANCE"

    def __init__(self, normalized_size: int = 40, luminance_size: int = 16,
                 chromatic_size: int = 12, gradient_size: int = 12,
                 occupancy_size: int = 12):
        self.normalized_size = int(normalized_size)
        self.luminance_size = int(luminance_size)
        self.chromatic_size = int(chromatic_size)
        self.gradient_size = int(gradient_size)
        self.occupancy_size = int(occupancy_size)

        a = self.luminance_size ** 2
        b = a + 2 * self.chromatic_size ** 2
        c = b + 3 * self.gradient_size ** 2
        d = c + self.occupancy_size ** 2
        # 12 robust global moments stabilize matching when the raster changes
        # slightly under crop/viewpoint variation.
        e = d + 12
        self.layout = AppearanceLayout(e, {
            "local_luminance": (0, a),
            "chromaticity": (a, b),
            "gradient_field": (b, c),
            "occupancy": (c, d),
            "global_moments": (d, e),
        })

    @property
    def dim(self) -> int:
        return self.layout.dim

    @staticmethod
    def _safe_mask(image: np.ndarray, attention_mask: Optional[np.ndarray], bbox) -> np.ndarray:
        h, w = image.shape[:2]
        if attention_mask is not None:
            m = attention_mask[..., 0] if attention_mask.ndim == 3 else attention_mask
            m = (m > 0).astype(np.uint8) * 255
            if int(np.count_nonzero(m)) >= 8:
                return m
        if bbox is None:
            return np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [float(v) for v in bbox]
        x0 = int(np.clip(round(x*w), 0, max(0, w-1)))
        y0 = int(np.clip(round(y*h), 0, max(0, h-1)))
        x1 = int(np.clip(round((x+bw)*w), x0+1, w))
        y1 = int(np.clip(round((y+bh)*h), y0+1, h))
        m = np.zeros((h,w), dtype=np.uint8); m[y0:y1,x0:x1] = 255
        return m

    @staticmethod
    def _principal_angle(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return 0.0, 0.0
        x = xs.astype(np.float64) - xs.mean(); y = ys.astype(np.float64) - ys.mean()
        cov = np.asarray([[np.mean(x*x), np.mean(x*y)], [np.mean(x*y), np.mean(y*y)]])
        vals, vecs = np.linalg.eigh(cov); order = np.argsort(vals)[::-1]
        vals = vals[order]; v = vecs[:,order[0]]
        anisotropy = float((vals[0]-vals[1]) / max(vals[0]+vals[1], 1e-9))
        angle = float(np.degrees(np.arctan2(v[1],v[0])))
        return angle, anisotropy

    def _prepare(self, image: np.ndarray, attention_mask: Optional[np.ndarray], bbox=None):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 BGR image")
        mask = self._safe_mask(image, attention_mask, bbox)
        work_i = np.ascontiguousarray(image); work_m = mask

        # Pose-normalize only anisotropic regions. The 180-degree ambiguity is
        # intentionally left to the bounded multi-view prototype bank.
        angle, anis = self._principal_angle(mask)
        if anis >= .12:
            ys,xs = np.where(mask>0); center=(float(xs.mean()),float(ys.mean()))
            rot=cv2.getRotationMatrix2D(center, angle, 1.0)
            bg=tuple(int(v) for v in np.median(image.reshape(-1,3),axis=0))
            work_i=cv2.warpAffine(image,rot,(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=bg)
            work_m=cv2.warpAffine(mask,rot,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=0)

        ys,xs=np.where(work_m>0)
        if len(xs)<8:
            raise ValueError("focused region contains too few pixels")
        x0,x1=int(xs.min()),int(xs.max())+1; y0,y1=int(ys.min()),int(ys.max())+1
        pad=max(1,int(round(.04*max(x1-x0,y1-y0))))
        x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(work_i.shape[1],x1+pad); y1=min(work_i.shape[0],y1+pad)
        ci=work_i[y0:y1,x0:x1]; cm=work_m[y0:y1,x0:x1]
        h,w=cm.shape; side=max(h,w)
        bg=np.median(image.reshape(-1,3),axis=0).astype(np.uint8)
        si=np.empty((side,side,3),dtype=np.uint8); si[:]=bg
        sm=np.zeros((side,side),dtype=np.uint8)
        oy,ox=(side-h)//2,(side-w)//2
        si[oy:oy+h,ox:ox+w]=ci; sm[oy:oy+h,ox:ox+w]=cm
        n=self.normalized_size
        ni=cv2.resize(si,(n,n),interpolation=cv2.INTER_AREA)
        nm=cv2.resize(sm,(n,n),interpolation=cv2.INTER_AREA)
        return ni,nm

    @staticmethod
    def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> Tuple[float,float]:
        w=np.asarray(weights,dtype=np.float64).reshape(-1); x=np.asarray(values,dtype=np.float64).reshape(-1)
        s=max(float(w.sum()),1e-9); mean=float(np.sum(w*x)/s)
        var=float(np.sum(w*(x-mean)**2)/s)
        return mean, float(np.sqrt(max(var,0.0)))

    def extract(self, image: np.ndarray, *, attention_mask: Optional[np.ndarray]=None, bbox=None) -> np.ndarray:
        ni,nm=self._prepare(image,attention_mask,bbox)
        img=ni.astype(np.float64)/255.0; mask=np.clip(nm.astype(np.float64)/255.0,0,1)
        focus=mask>0.12
        if not np.any(focus): focus=np.ones_like(mask,dtype=bool)

        # Local-contrast luminance suppresses brightness shifts while retaining
        # fine markings/layout.
        gray=cv2.cvtColor(ni,cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
        vals=gray[focus]; gmean=float(vals.mean()); gstd=max(float(vals.std()),.045)
        local=np.clip((gray-gmean)/(2.8*gstd),-1,1)
        local=(local+1.0)*.5
        lum=cv2.resize(local,(self.luminance_size,self.luminance_size),interpolation=cv2.INTER_AREA).reshape(-1)

        # Chromaticity removes overall brightness. B/G are sufficient because
        # R=1-B-G. These are generic channel relationships, not hue labels.
        f=img+1e-4; denom=np.maximum(f.sum(axis=2),1e-8)
        c0=f[:,:,0]/denom; c1=f[:,:,1]/denom
        c0=cv2.resize(c0,(self.chromatic_size,self.chromatic_size),interpolation=cv2.INTER_AREA).reshape(-1)
        c1=cv2.resize(c1,(self.chromatic_size,self.chromatic_size),interpolation=cv2.INTER_AREA).reshape(-1)

        gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        mag=np.sqrt(gx*gx+gy*gy); scale=max(float(np.quantile(mag[focus],.90)) if np.any(focus) else .1,.04)
        mag=np.clip(mag/scale,0,1)
        denomg=np.maximum(np.sqrt(gx*gx+gy*gy),1e-8)
        ux=np.clip(gx/denomg,-1,1); uy=np.clip(gy/denomg,-1,1)
        gr=[]
        for a in (mag,(ux+1)*.5,(uy+1)*.5):
            gr.append(cv2.resize(a,(self.gradient_size,self.gradient_size),interpolation=cv2.INTER_AREA).reshape(-1))
        grad=np.concatenate(gr)

        occ=cv2.resize(mask,(self.occupancy_size,self.occupancy_size),interpolation=cv2.INTER_AREA).reshape(-1)

        moments=[]
        weights=mask
        for ch in range(3):
            m,s=self._weighted_stats(img[:,:,ch],weights); moments.extend([m,s])
        for a in (gray,c0.reshape(self.chromatic_size,self.chromatic_size) if False else img[:,:,0]/np.maximum(img.sum(axis=2),1e-8),
                  img[:,:,1]/np.maximum(img.sum(axis=2),1e-8)):
            m,s=self._weighted_stats(a,weights); moments.extend([m,s])
        moments=np.asarray(moments[:12],dtype=np.float64)

        vec=np.concatenate([lum,c0,c1,grad,occ,moments]).astype(np.float64)
        if vec.shape != (self.dim,):
            raise RuntimeError(f"unexpected V0.13 appearance dimension {vec.shape}, expected {(self.dim,)}")
        return vec

    def summary(self) -> Dict[str,object]:
        return {"version":self.VERSION,"feature_dim":self.dim,"groups":dict(self.layout.groups),
                "learned_weights":0,"raw_images_retained":0}

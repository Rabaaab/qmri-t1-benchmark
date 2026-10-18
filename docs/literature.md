# Literature search

Record what you searched, where, and what you found. An examiner will ask
"how do you know nobody has done this?" and the only acceptable answer is a
record like this one, not a memory.

Fill in the **Found** column as you go. "Nothing relevant" is a real result
and worth writing down — it is evidence you looked.

---

## A. The direct hit

If someone has already compared a CNN against voxel-wise fitting for VFA T1
mapping, you want to know **now**. This is good news if you find it: you
replicate, extend, and cite. It is only bad news if you find it in November.

**PubMed**

```
("T1 mapping"[tiab] OR "T1 relaxometry"[tiab] OR DESPOT1[tiab] OR "variable flip angle"[tiab]) AND ("deep learning"[tiab] OR "neural network"[tiab] OR convolutional[tiab])
```

```
("variable flip angle"[tiab] OR VFA[tiab]) AND T1[tiab] AND (bias[tiab] OR accuracy[tiab]) AND ("machine learning"[tiab] OR "deep learning"[tiab])
```

**Google Scholar**

```
"variable flip angle" "T1 mapping" ("deep learning" OR "neural network") bias
```

```
DESPOT1 ("deep learning" OR CNN OR "U-Net") T1 estimation
```

| Query | Date | Hits | Found |
|---|---|---|---|
| | | | |

---

## B. The headline: do networks erase small structures?

This is your actual claim. Note it lives in **two** separate literatures and
the vocabulary differs — that is why it is easy to miss.

In *reconstruction* the words are hallucination, instability, false negative.
In *parameter estimation* the words are bias, training distribution, sink.
Search both.

**Google Scholar**

```
"deep learning" MRI reconstruction hallucination pathology lesion
```

```
deep learning MRI "small lesion" (removed OR missed OR "smoothed") reconstruction
```

```
fastMRI challenge hallucination pathology Muckley
```

```
Antun instabilities deep learning image reconstruction
```

```
"lesion conspicuity" deep learning quantitative MRI
```

**PubMed**

```
("deep learning"[tiab] OR "neural network"[tiab]) AND MRI[tiab] AND (hallucination[tiab] OR conspicuity[tiab] OR detectability[tiab]) AND (lesion[tiab] OR pathology[tiab])
```

| Query | Date | Hits | Found |
|---|---|---|---|
| | | | |

---

## C. Bias in ML parameter estimation (cite regardless)

You must cite this whatever you find. The anchor is:

> Gyori NG, Palombo M, Clark CA, Zhang H, Alexander DC.
> "Training data distribution significantly impacts the estimation of tissue
> microstructure with machine learning." *Magn Reson Med.* 2022;87(2):932-947.

**Highest-yield single action in this whole document:** open that paper in
Google Scholar and click **"Cited by"**. Everything downstream of it is
directly relevant, and it is a much better filter than any keyword query.
Then do the same in reverse — read its reference list.

**Google Scholar**

```
"training data distribution" parameter estimation quantitative MRI bias
```

```
("machine learning" OR "deep learning") "quantitative MRI" "bias" "variance" parameter estimation
```

```
neural network parameter estimation "regression to the mean" OR "regression toward the mean" MRI
```

| Query | Date | Hits | Found |
|---|---|---|---|
| | | | |

---

## D. Justify the numbers in this repo

Every constant in `src/qmri/` was chosen by plausibility, not by reading.
Some are probably wrong. Optimal flip-angle selection for VFA is a *solved*
problem and this project has not consulted the solution.

**Google Scholar**

```
Deoni optimal flip angle DESPOT1 T1 precision SNR
```

```
"variable flip angle" T1 mapping "optimal" flip angle selection precision
```

```
B1 inhomogeneity correction "variable flip angle" T1 mapping brain 3T
```

```
Rician noise bias T1 estimation low SNR magnitude MRI
```

Specific papers to find and read:

- Deoni SCL, Rutt BK, Peters TM. Rapid combined T1 and T2 mapping using
  gradient recalled acquisition in the steady state. *MRM* 2003.
- Chang LC, Koay CG, Basser PJ, Pierpaoli C. Linear least-squares method for
  unbiased estimation of T1 from SPGR signals. *MRM* 2008;60(2):496-501.
- Wansapura JP, et al. NMR relaxation times in the human brain at 3.0 Tesla.
  *JMRI* 1999;9(4):531-538.
- Stanisz GJ, et al. T1, T2 relaxation and magnetization transfer in tissue
  at 3T. *MRM* 2005;54(3):507-512.

| Question | Current value in repo | Justified by | Changed to |
|---|---|---|---|
| number of flip angles | 5 | | |
| flip angles | 2, 5, 10, 15, 20 deg | | |
| TR | 15 ms | | |
| SNR | 100 | | |
| B1 amplitude | 0.15 | | |
| lesion T1 | 1200 ms | | |
| lesion sizes | 2-12 px across | | |

---

## E. Where else to look

**ISMRM abstracts.** A large fraction of qMRI methods work appears as
conference abstracts and never becomes a paper. PubMed will not find them.
Search `cds.ismrm.org`, and also try Google with `site:cds.ismrm.org`.

**arXiv.** Methods work often appears there months before publication.
Search `arxiv.org` for `T1 mapping deep learning bias`.

**qMRLab** (`qmrlab.org`) — an open-source qMRI toolbox from NeuroPoly, the
same lab as the supervisor. Worth knowing in detail before any interview:
it tells you what that group considers standard practice.

---

## How to judge what you find

**Green flag — strengthens the project.** Someone shows ML qMRI estimators
are biased, in a different modality or with voxel-wise models. That is
support, it is citable, and it leaves your spatial-context and
lesion-size-resolved angle intact.

**Yellow flag — narrows the project.** Someone did a CNN for VFA T1 but did
not measure lesion retention. Then retention is your contribution and the
rest is replication. Still fine, and now better positioned.

**Red flag — change the angle.** Someone did exactly this: CNN vs voxel-wise
for VFA T1, with lesion-size-resolved retention. Then pivot, and say so
early. Options if it happens: change the parameter (T2, ADC), change the
failure mode (out-of-distribution B1 instead of lesions), or replicate their
result and test whether it survives a condition they did not try.

Finding a red flag in October costs you a week. Finding it in December costs
you the project.

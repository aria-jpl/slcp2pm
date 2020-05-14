### slcp2pm (LAR)

#### SLC Pair to Log Amplitude Ratio

This repository contains jobs to convert S1-SLCP products to Log Amplitude Ratios (LARs) for creating Flood Proxy Maps v1 (FPMv1).

A LAR is created based on the below pairing system, using 2 SLCs --> 1 SLCP --> 1 FPM:
![image](https://user-images.githubusercontent.com/6346909/77985672-9f6cea00-7304-11ea-807d-861833d4b1a3.png)

_Note: Since SLCP:LAR is 1:1, the pairing of SLCPs is **not required** to make LARs. Hence, user simply needs to facet on target SLCPs to execute the LAR job on._


### Job 1: S1 Log Amplitude Ratio
- Type: **Iterative**
- Facet: **SLCPs to create LARs from**
- User inputs:

    | Fields        | Description   | Type  |Example  |
    | ------------- |-------------| :---------:| :---------:|
    | `lar_range_looks` | Range looks to create LARs. (Overrides looks in SLCP's metadata)|  int | 7 |
    | `lar_azimuth_looks` | Azimuth looks to create LARs. (Overrides looks in SLCP's metadata) |  int | 2 |

- Important outputs:

    | Product        | Description   | Example  |
    | ------------- |-------------| :-----|
    | Log Amplitude Ratio Products | Geocoded, multilooked LARs   | 	logr_[burst]_[range_lks]_[az_lks].float.geo|
    | Amplitude Products | Geocoded, multilooked amplitudes stored in 2 bands. Band 1 - Master scene's amp. Band 2 - Slave scene's amp.  | 	amp_[subswath]_[range_lks]_[az_lks].amp.geo|


##### Notes on S1 Log Amplitude Ratio
The LARs in this PGE are computed as such (from `log_ratio.py`):

* (**Latest**) From dataset `v2.0` onwards:

    ![formula](https://render.githubusercontent.com/render/math?math=LAR=\log_{10}{\frac{A_{post-event}}{A_{pre-event}}})
    ![formula](https://render.githubusercontent.com/render/math?math==\log_{10}{\frac{A_{slave}}{A_{master}}})
    
    where _A_ = Amplitudes of SLCs of the given date in the co-registered SLCPs
    
    => Negative values / darker pixels correspond to decreased amplitudes in the post-event scene and possible open-water flood.

* Before dataset `v2.0` (`v1.x` etc):

    ![formula](https://render.githubusercontent.com/render/math?math=LAR=\log_{10}{\frac{A_{pre-event}}{A_{post-event}}})
    ![formula](https://render.githubusercontent.com/render/math?math==\log_{10}{\frac{A_{master}}{A_{slave}}})
    
    where _A_ = Amplitudes of SLCs of the given date in the co-registered SLCPs
    
    => Positive values / brighter pixels correspond to decreased amplitudes in the post-event scene and possible open-water flood.


### How to use the raw code for standalone use

**How to create log amplitude ratio images from ARIA SLC_PAIR products:**

1. Go to src.
```
   ./compile.sh
```
2. Set environment variables. See set_env_variable.sh for details.
3. Then modify slcp2lar_S1.sh and run it.

### Authors

* **Sang-Ho Yun** - *initial work*

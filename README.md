
# Spatio-Temporal Dynamic Inference Network for Group Activity Recognition

The source codes for ICCV2021 Paper: 
Spatio-Temporal Dynamic Inference Network for Group Activity Recognition.  
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Spatio-Temporal_Dynamic_Inference_Network_for_Group_Activity_Recognition_ICCV_2021_paper.pdf)
[[supplemental material]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Yuan_Spatio-Temporal_Dynamic_Inference_ICCV_2021_supplemental.pdf)
[[arXiv]](http://arxiv.org/abs/2108.11743)

[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark?style=social)](https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark)](https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JacobYuan7/DIN-Group-Activity-Recognition-Benchmark)

If you find our work or the codebase inspiring and useful to your research, please cite
```bibtex
@inproceedings{yuan2021DIN,
  title={Spatio-Temporal Dynamic Inference Network for Group Activity Recognition},
  author={Yuan, Hangjie and Ni, Dong and Wang, Mang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7476--7485},
  year={2021}
}
@inproceedings{yuan2021visualcontext,
  title={Learning Visual Context for Group Activity Recognition},
  author={Yuan, Hangjie and Ni, Dong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3261--3269},
  year={2021}
}
```
        


## Dependencies

- Software Environment: Linux (CentOS 7)
- Hardware Environment: NVIDIA TITAN RTX
- Python `3.6`
- PyTorch `1.2.0`, Torchvision `0.4.0`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)



## Prepare Datasets

1. Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
2. Unzip the dataset file into `data/volleyball` or `data/collective`.
3. Download the file `tracks_normalized.pkl` from [cvlab-epfl/social-scene-understanding](https://raw.githubusercontent.com/wjchaoGit/Group-Activity-Recognition/master/data/volleyball/tracks_normalized.pkl) and put it into `data/volleyball/videos`


## Using Docker
1. Checkout repository and `cd PROJECT_PATH`

2. **Build the Docker container**
```shell
docker build -t din_gar https://github.com/JacobYuan7/DIN_GAR.git#main
```

3. **Run the Docker container**
```shell
docker run --shm-size=2G -v data/volleyball:/opt/DIN_GAR/data/volleyball -v result:/opt/DIN_GAR/result --rm -it din_gar
```
- `--shm-size=2G`: To prevent _ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)._, you have to extend the container's shared memory size. Alternatively: `--ipc=host`
- `-v data/volleyball:/opt/DIN_GAR/data/volleyball`: Makes the host's folder `data/volleyball` available inside the container at `/opt/DIN_GAR/data/volleyball`
- `-v result:/opt/DIN_GAR/result`: Makes the host's folder `result` available inside the container at `/opt/DIN_GAR/result`
- `-it` & `--rm`: Starts the container with an interactive session (PROJECT_PATH is `/opt/DIN_GAR`) and removes the container after closing the session.
- `din_gar` the name/tag of the image
- optional: `--gpus='"device=7"'` restrict the GPU devices the container can access.

## Get Started
1. **Train the Base Model**: Fine-tune the base model for the dataset. 
    ```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
    ```

2. **Train with the reasoning module**: Append the reasoning modules onto the base model to get a reasoning model.
    1. **Volleyball dataset**
        - **DIN** 
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
       - **lite DIN** \
            We can run DIN in lite version by setting *cfg.lite_dim = 128* in *scripts/train_volleyball_stage2_dynamic.py*.
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
       - **ST-factorized DIN** \
            We can run ST-factorized DIN by setting *cfg.ST_kernel_size = [(1,3),(3,1)]* and *cfg.hierarchical_inference = True*.
        
            **Note** that if you set *cfg.hierarchical_inference = False*, *cfg.ST_kernel_size = [(1,3),(3,1)]* and *cfg.num_DIN = 2*, then multiple interaction fields run in parallel.
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
        
        Other model re-implemented by us according to their papers or publicly available codes:
        - **AT** 
            ```
            python scripts/train_volleyball_stage2_at.py
            ```
        - **PCTDM** 
            ```
            python scripts/train_volleyball_stage2_pctdm.py
            ```
        - **SACRF** 
            ```
            python scripts/train_volleyball_stage2_sacrf_biute.py
            ```
       - **ARG** 
            ```
            python scripts/train_volleyball_stage2_arg.py
            ```
        - **HiGCIN** 
            ```
            python scripts/train_volleyball_stage2_higcin.py
            ```
       
    2. **Collective Activity dataset**
        -  **DIN** 
            ```
            python scripts/train_collective_stage2_dynamic.py
            ```
        -  **DIN lite** \
        We can run DIN in lite version by setting 'cfg.lite_dim = 128' in 'scripts/train_collective_stage2_dynamic.py'.
            ```
            python scripts/train_collective_stage2_dynamic.py
            ```

Another work done by us, solving GAR from the perspective of incorporating visual context, is also [available](https://ojs.aaai.org/index.php/AAAI/article/view/16437/16244).
```bibtex
@inproceedings{yuan2021visualcontext,
  title={Learning Visual Context for Group Activity Recognition},
  author={Yuan, Hangjie and Ni, Dong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3261--3269},
  year={2021}
}
```








# Referencing Where to Focus: Improving Visual Grounding with Referential Query

Welcome to the official repository for the method presented in "[Referencing Where to Focus: Improving Visual Grounding with Referential Query](https://proceedings.neurips.cc/paper_files/paper/2024/file/54c67d3db2df24a31cf045525f9460b9-Paper-Conference.pdf)".

RefFormer consists of the query adaption module that can be seamlessly integrated into CLIP and generate the referential query to provide the prior context for decoder, along with a task-specific decoder. By incorporating the referential query into the decoder, we can effectively mitigate the learning difficulty of the decoder, and accurately concentrate on the target object. Additionally, our proposed query adaption module can also act as an adapter, preserving the rich knowledge within CLIP without the need to tune the parameters of the backbone network.


![image](framework.png)

### Dataset
- RefCOCO/+/g
- Flickr30K
- ReferItGame



### Training

```
sh run_grounding.sh
```



## Reference

If you find the package useful, please consider citing our paper:

```
@article{wang2025referencing,
  title={Referencing Where to Focus: Improving Visual Grounding with Referential Query},
  author={Wang, Yabing and Tian, Zhuotao and Guo, Qingpei and Qin, Zheng and Zhou, Sanping and Yang, Ming and Wang, Le},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={47378--47399},
  year={2025}
}
```






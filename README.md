# GearNet
 AAAI 2022: GearNet: Stepwise Dual Learning for Weakly Supervised Domain Adaptation. (Pytorch implementation)

## Requirements
* Python 3.8.3
* PyTorch 1.6.0
## Taining
All the commands are shown in `scripts` files.

##Results
| UNIF-20% | A -> W | A -> D |W -> A | W -> D |D -> A |D -> W | Average |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Standard |0.6197|	0.6312	| 0.4861 |	0.9229 |	0.3164 |	0.8085 |	0.6308 |
| Co-teaching | 0.6158	| 0.6791	| 0.5614	| 0.9437 |	0.5482|	0.845|	0.6988|
| Jocor | 0.6653	| 0.7416	| 0.5237	| 0.9770	| 0.5028	| 0.9036	| 0.7190|
| DAN | 0.7031	 |0.7187	 |0.5809 |	0.9562	 |0.6299 |	0.9127 |	0.75025 |
| DANN | 0.7122|	0.7500|	0.6093|	0.9687|	0.6157	|0.9270	|0.7638 |
| TCL | 0.7773	|0.8166	|0.6075| 0.9812| 	0.6100 |	0.9361|	0.7881 |
| G+Co-teaching | 0.7343|	0.7791|	0.6058	|0.9500|	0.6051	|0.9036|	0.7629 |
| G+DANN | 0.7552|	0.7916|	0.6175|	0.9729|	0.6139|	0.9361	|0.7812 |
| G+TCL | 0.8177	|0.8437|	0.6168|	0.9875|	0.6225	|0.9518	|0.8066|

## Citation
If you find this useful in your research, please consider citing:
```
@article{xie2022gearnet,
  title={GearNet: Stepwise Dual Learning for Weakly Supervised Domain Adaptation},
  author={Xie, Renchunzi and Wei, Hongxin and Feng, Lei and An, Bo},
  journal={AAAI},
  year={2022}
}
```

## Contact
If you have any problems about our code, feel free to contact<br>

* XIER0002@e.ntu.edu.sg

or describe your problem in Issues.

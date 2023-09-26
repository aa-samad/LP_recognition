# RILP
### Robust Iranian License Plate Recognition Designed for Complex Conditions

**This Repo is not maintaned anymore, there are issues with the code and the dataset is not available anymore for download. However, please adequately cite the paper if you compare to our method or your method is built using our code/method backbone.**

This is a Modular Framework designed for License Plate Number Recognition in Complex Conditions.
The distinct Design ables us to reconfigure the framework for other regions/Conditions in no time!

![img](Demo.png)

## Prerequisites
The Following Setup is tested and working:
- Python>=3.5
- Pytorch>=0.4.1
- Tensorflow>=1.12.2
- Cuda>=9.0
- opencv>=3.4.2

## Testing
- Place the images inside **test_set/images** directory
- Delete all other images inside folders (don't delete the folders, just files inside them)
- In main directory run: ```python3 runner.py```

## Training
- The pre-trained model provided, we will not publish the training code
- In order to train use pre-trained model or try another model

## datasets
### Glyphs
![img1](glyph_ex.jpg)

- Properties:
	- Volume: 5000 images
	- Labeled
	- Size: 100 x K  (20 < K < 100)
- Link:
	- Will be avaiable soon, stay tuned.

### Plates
![img2](dataset_ex.jpg)

- Properties:
	- Volume: 350 images
	- Various conditions
	- Various sizes
	- License plate number (only one of them) is labeled
- Link:
	- It is only available for research purposes. Ask the first author a_samad[at]aut[dot]ac[dot]ir
## Citing
Please adequately refer to the papers any time this Method/Code is being used. If you do publish a paper where this Work helped your research, Please cite the following papers in your publications.

	@inproceedings{Samadzadeh2020RILP,
	  title={RILP: Robust Iranian License Plate Recognition Designed for Complex Conditions},
	  author={Ali Samadzadeh, Amir Mehdi Shayan, Bahman Rouhani, Ahmad Nickabadi, Mohammad Rahmati},
	  year={2020},
	  organization={IEEE}}

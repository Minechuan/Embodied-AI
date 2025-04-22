



# Test and Evaluate Helper

## Test

### Part 1: Directly Estimate the Object Pose

```
python test.py --checkpoint=exp_checkpoint/part1_directly/checkpoint/part1_checkpoint.pth --mode=val --device=cuda
```
one of my test result: Error: trans 0.01709689013659954 rot 0.2098878192612525

### part 2: Estimate the Object Pose by Fitting (With RANSAC parallel)

```
python test.py --checkpoint=exp_checkpoint\part2_fitting\checkpoint\coord_fitting.pth --mode=val --device=cuda
```
K 100, inlier_thresh= 0.03
one of my test result: Error: trans 0.0015633522998541594 rot 0.025049367889629325

## eval

### Part 1: Directly Estimate the Object Pose
```
python eval.py --checkpoint=exp_checkpoint/part1_directly/checkpoint/part1_checkpoint.pth --mode=val --device=cuda --vis=0 --headless=1
```
one of my test result: Current success rate: 459/500 = 0.918

### part 2: Estimate the Object Pose by Fitting (With RANSAC parallel)
```
python eval.py --checkpoint=exp_checkpoint\part2_fitting\checkpoint\coord_fitting.pth --mode=val --device=cuda --vis=0 --headless=1 
```
one of my test result: Current success rate: 484/500 = 0.968
# About
>This project uses code from [https://github.com/cyh24/Joint-Bayesian](https://github.com/cyh24/Joint-Bayesian). Using LBP feature has a better performance than using CNN, because we only use LFW dataset to train the Joint-Bayesian model, I think this algorithm can't get good model when use a small dataset.

# CNN
>We use google net to train the model, the dataset is CASIA Webface, and we use [Effective Face Frontalization in Unconstrained Images](http://arxiv.org/abs/1411.7964) to do alignment, we used PLDA to do the experiment(It's performance is better than using Joint Bayesian in this code, however the PLDA algorithm is writen using Matlab). Here is our result using [BLUFR protocol](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/):

>Acc = 0.9273

>@ FAR = 0.1%: VR = 68.63%.
	
>@ FAR = 1%: VR = 83.78%.
	
>The performance is not so good, using better dataset will be helpful, we get 97% accuracy in LFW when we use a extended CASIA dataset, but I can't get these data and models now. I will try to train new model in the future.

# Joint-Bayesian
>According to the paper *"Bayesian Face Revisited: A Joint Formulation"*, the repository realizes the algorithm of Joint Beyesian with **Python** and achieve almost the same result as the paper.

# Prepare for Using
 >1. Get the database (lbp_WDRef,id_WDRef,lbp_lfw,pairlist_lfw)
 Download from the Websit: http://home.ustc.edu.cn/~chendong/JointBayesian/
 >2. Install the numpy & scipy
 >3. Install the sklearn
 >4. If you want to use CNN to extract features, please install pycaffe and train a model
 >5. You need to change some path and filename in code, I think the code can explain itself.
 

# Usage
>cd src

>python test_lfw.py

# More information
>You can get more information in my [blog](http://lufo.me/2015/11/face_verification_demo/)
>If you have any question, my email: lufo816@gmail.com

GAN for FER

Only one Discriminator for image and feature!
(只用一个判别器同时学习image和image feature)

main.py中的class GAN和class Train分别参考[GANimation](https://github.com/albertpumarola/GANimation)中的class GANimation和class Train


main.py (z: [bs, z_dim], is feature vector)


	  	   img_real  ->  E  ->  z_real
 	
		(z_real, y)  ->  G  ->  img_fake
       	       (z_prior, y)  ->  G  ->  img_fake_from_prior
	
	(img_real, z_prior)  ->  D  ->  (logits_gan, logits_cls)    1
	 (img_real, z_real)  ->  D  ->  (logits_gan, logits_cls)    0
	(img_fake, z_prior)  ->  D  ->  (logits_gan, logits_cls)    0




maingz.py (gx: [bs, c, h, w], is feature map)

	  	   img_real  ->  E  ->  gx_real
       		    z_prior  ->  Gz ->  gx_fake
 	
	       (gx_real, y)  ->  G  ->  img_fake
       	       (gx_fake, y)  ->  G  ->  img_fake_from_prior
	
	(img_real, gx_real)  ->  D  ->  (logits_gan, logits_cls)    1
	(img_real, gx_fake)  ->  D  ->  (logits_gan, logits_cls)    0
	(img_fake, gx_real)  ->  D  ->  (logits_gan, logits_cls)    0


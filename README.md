# Breast_Cancer_Dissertation
Breast cancer death rates have been decreasing steadily since 1989[1], nonetheless, this type
of cancer is the second leading cause of cancer deaths among women worldwide. This de-
crease is a result of an early cancer diagnosis with the help of screening processes, therefore
the current efforts are focused on reducing false positive and false negative results and hence,
decreasing mortality. Newer and experimental breast imaging techniques have surged in an
attempt to decrease mortality rates, one of these new techniques is called contrast-enhanced
spectral mammography (CESM) which has already shown improvements in breast cancer diag-
nosis, nevertheless, there is no trace of using this new imaging technique in combination with a
deep learning algorithm to further improve detection and diagnosis performance. Two models
were designed and implemented, the first was trained using the U-Net architecture as a basis
to segment lesions in CESM images, and the second one was trained on a convolutional neural
network (CNN) model in order to classify breast lesions as either benign, malignant or nor-
mal. Both models achieved great results with a 100% sensitivity and 79% specificity in the case
of breast cancer classification and a 95% Dice score and 94.5% intersection over union (IoU)
for the segmentation model which not only surpassed state-of-the-art results but also exceeded
expected improvements.


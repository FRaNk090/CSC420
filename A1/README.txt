To run the program, please cd to the directory first. Then just simply run

python3 a1_code.py 
or 
python a1_code.py

in the terminal.

If error ouccurs, try things listed below:
1. Please make sure that libraries like cv2, plt, numpy are up tp date.
2. please try updating python version to 3.8 or newer.
3. Also check if there is folder called 'A1_images' under the same directory.
   The images in A1_images should be image1.jpg, image2.jpg, image3.jpg


Note that:
1. In lines 76 -78, you can vary the parameter values test the functionalities
2. The shapes of images in each step are printed out in terminal.
3. If run successfully, it will show 1 gaussian filter image and 4 plots for each image in the A1_images folder. 13 in total.
4. There are 4 columns:
    i) The original image.
    ii) The image after taking gaussian blur.
    iii) The image after taking gradient magnitude.
    ix) The image after applying threshold.
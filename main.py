'''
This will be main file which the co-ordinaters of the event will be using to test your
code. This file contains two functions:

1. predict: You will be given an rgb image which you will use to predict the output 
which will be a string. For the prediction you can use/import code,models from other files or
libraries. More detailes given above the function defination.

2. test: This will be used by the co-ordinators to test your code by giving sample 
inputs to the 'predict' function mentioned above. A sample test function is given for your
reference but it is subject to minor changes during the evaluation. However, note that
there won't be any changes in the input format given to the predict function.

Make sure all the necessary functions etc. you import are done from the same directory. And in 
the final submission make sure you provide them also along with this script.
'''

# Essential libraries and your model can be imported here
import os
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
import segment


characters = ["क","घ","च","ज","ट","ठ","ढ","थ","प","फ","ब","म","र","ल","ष","स","ह","क्ष","त्र","ज्ञ","ॠ"]
image_size = 32

'''
function: predict
input: image - A numpy array which is an rgb image
output: answer - A list which is the full word

Suggestion: Try to make your code understandable by including comments and organizing it. For 
this we encourgae you to write essential function in other files and import them here so that 
the final code is neat and not too big. Make sure you use the same input format and return 
same output format.
'''
def predict(image):
    '''
    Write your code for prediction here.
    '''
    
    # Write the location of Model here
    model_path = "model.h5"
    Model = load_model(model_path)

    # answer = ['क','ख','ग'] # sample needs to be modified
    answer = []
    # path = "captcha.jpg"
    # original_img = cv2.imread(path)
    original_img = image
    
    # cv2_imshow("original_img",original_img)
        
    # Find all contours of words present in image if any present
    # This is special factor of our model that it can also read if multiple words given that they are written in same line.
    # You can see the sample images in "image" folder with name "INNOVATIVE_FEUTURE_3" and similar for more clearity
    final_contours = segment.find_contours_word(image)
    images = []
    # removes shadow in image
    original_img = segment.shadow_remove(original_img)
    # making white border to be on safer side at time of detection
    original_img = cv2.copyMakeBorder(original_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    
    for contour in final_contours:
        [x, y, w, h] = contour
        img = original_img[y : y + h, x : x + w]
        images.append(img)
    
    for img in images:
        # cv2_imshow("img**",img)
        letters = segment.segmentation(img)
        
        for image in letters:
            segment.cv2_imshow("image",image)
            gray_img = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_AREA)
            gray_img = np.array(gray_img)
            gray_img = gray_img.astype('float32')
            gray_img /= 255 
            gray_img.resize((1, 32, 32, 1))
            
            pred = Model.predict(gray_img)
            answer.append(characters[np.argmax(pred, axis=1)[0]])
    return answer


'''
function: test
input: None
output: None

This is a sample test function which the co-ordinaors will use to test your code. This is
subject to change but the imput to predict function and the output expected from the predict
function will not change. 
You can use this to test your code before submission: Some details are given below:
image_paths : A list that will store the paths of all the images that will be tested.
correct_answers: A list that holds the correct answers
score : holds the total score. Keep in mind that scoring is subject to change during testing.

You can play with these variables and test before final submission.
'''
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ["./images/test6.jpeg"]
    # image_paths = ['./image1','./image2',',./imagen']
    correct_answers = [["क","ल","म","र","ब","ज्ञ"]]
    # correct_answers = [list1,list2,listn]
    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(''.join(answer))# will be the output string

        n=0
        for j in range(len(answer)):
            if j < len(correct_answers[i]):
                if correct_answers[i][j] == answer[j]:
                    n+=1
                
        if(n==len(correct_answers[i])):
            score += len(correct_answers[i])*multiplication_factor

        else:
            score += n*2
        
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()
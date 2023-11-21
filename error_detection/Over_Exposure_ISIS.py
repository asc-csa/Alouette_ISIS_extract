#Jeyshinee P. Nov 2023 - Flagging Over Exposed ISIS Ionograms

#imports 
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
import os


def flag_overexposed(image_path, plotting_hist = False):
    '''
    Definition:
        flag_exposed takes in one ionogram and saves it if it is overexposed
    
    Parameters:
        image_path (str) : Path to image

        plotting_hist (bool) : Default= False. If true, display histogram plot 

    Return:
        saves over_exposed image 
    
    '''
    try:
        #Get frequency and bins for histogram

        path = image_path
        img = imageio.imread(path)
        image_intensity = img_as_ubyte(rgb2gray(img))
        freq, bins = histogram(image_intensity)
        width, height = img.shape[0], img.shape[1]
        total_pixels = width*height
       
       #Plot histogram, if true
        if plotting_hist:
            plt.step(bins, freq*1.0/freq.sum())
            plt.xlabel('intensity value')
            plt.ylabel('Fraction of pixels')
            plt.show()
        
        #Get histogram integral values for pixels >= 230
        integral_255 = 0
        for i in range(230,len(bins)-1):
            bin_width = bins[i+1] - bins[i]

            # Sum over number in each bin and multiply by bin width
            integral_255 = integral_255 + (bin_width * sum(freq[i:i+1]))
        print("for pixels = [200:" , str(len(bins) -1) , "], integral = ", integral_255)
        proportion = integral_255/total_pixels

       # to do: include OTSU threshold 

        if proportion > 0.11:
            os.chdir('C:/Users/jpyneeandee/Documents/Over_Exposure')
            plt.imshow(img)
            plt.savefig(str(proportion)+ '.png')

    except Exception as e:
        print("Error -", e)


#Manual Testing 

sub_directory = "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-260"

for filename in os.listdir(sub_directory):
    file_path = sub_directory + "/" + filename
    flag_overexposed(file_path)

#"L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-260" flagged 73 out of 385

over_exposed = ["L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-262/Image0085.png",
           "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-262/Image0003.png",
           "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-261/Image0259.png",
           "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-271/Image0001.png",
           "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-271/Image0003.png"]

normal_ionograms = [ "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-261/Image0260.png",
                     "L:/DATA/ISIS/ISIS_101300030772/b34_R014207854/B1-35-12 ISIS A C-1876/Image0092.png",
                     "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-260/Image0196.png",
                     "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-259/Image0261.png"]

film_issue = [
 "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1131/Image0003.png",
 "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-275/Image0479.png",
 "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1153/Image0869.png",
 "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1153/Image0870.png",
 "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-262/Image0003.png"]
 
 
for im in film_issue:
    flag_overexposed(im)
    
 
#overexposure test
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-262/Image0085.png"
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-262/Image0003.png"
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-261/Image0259.png" #hist not okay 
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-271/Image0001.png" #hist not okay
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-271/Image0003.png" # hist not okay

#normal test
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-261/Image0260.png" #200 does not pass
# "L:/DATA/ISIS/ISIS_101300030772/b34_R014207854/B1-35-12 ISIS A C-1876/Image0092.png" 
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-260/Image0196.png"
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-259/Image0261.png" 

# film issue test 
# "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1131/Image0003.png"
# "L:/DATA/ISIS/ISIS_101300030772/b7_R014207896/B1-34-50 ISIS A C-275/Image0479.png"
# "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1153/Image0869.png"
# "L:/DATA/ISIS/ISIS_101300030772/b18_R014207880/B1-35-32 ISIS B D-1153/Image0870.png"

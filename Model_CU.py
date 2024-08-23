import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import cv2
import sys, os

from copy import deepcopy
from unidecode import unidecode

import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from paddleocr import PaddleOCR

from JaroDistance import jaro_distance
from ProcessPDF import  binarized_image, get_checkboxes, get_iou
from ConditionFilter import condition_filter

MODEL = "CU"

if 'AppData' in sys.executable:
    application_path = os.getcwd()
else : 
    application_path = os.path.dirname(sys.executable)

OCR_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_config.json")
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))
MODEL_HELPER = OCR_HELPER[MODEL]
OCR_PATHES = OCR_HELPER["PATHES"]

CHECK_PATH = os.path.join(application_path, OCR_PATHES["checkboxes_path"][MODEL])

NULL_OCR = {"text" : "",
            "box" : [],
            "proba" : 0
           }

class KeyMatch:
    def __init__(self, seq_index, confidence, number_of_match, last_place_word, key_index, OCR):
        self.seq_index = seq_index
        self.confidence = confidence
        self.number_of_match = number_of_match
        self.last_place_word = last_place_word
        self.key_index = key_index
        self.OCR = OCR

class ZoneMatch:
    def __init__(self, local_OCR, match_indices, confidence, res_seq):
        self.local_OCR = local_OCR
        self.match_indices = match_indices
        self.confidence = confidence
        self.res_seq = res_seq

def paddle_OCR(image, show=False):

    def _cleanPaddleOCR(OCR_text):
        res = []
        for line in OCR_text:
            for t in line:
                    model_dict = {
                        "text" : "",
                        "box" : [],
                        "proba" : 0
                    }
                    model_dict["text"] = t[1][0]
                    model_dict["box"] = t[0][0]+t[0][2] #  x1,y1,x2,y2
                    model_dict["proba"] = round(t[1][1],3)
                    res.append(model_dict)
        
        return res

    def _order_by_tbyx(OCR_text):
        res = sorted(OCR_text, key=lambda r: (r["box"][1], r["box"][0]))
        for i in range(len(res) - 1):
            for j in range(i, 0, -1):
                if abs(res[j + 1]["box"][1] - res[j]["box"][1]) < 30 and \
                        (res[j + 1]["box"][0] < res[j]["box"][0]):
                    tmp = deepcopy(res[j])
                    res[j] = deepcopy(res[j + 1])
                    res[j + 1] = deepcopy(tmp)
                else:
                    break
        return res
    
    ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log = False) # need to run only once to download and load model into memory
    results = ocr.ocr(image, cls=True)
    results = _cleanPaddleOCR(results)
    results = _order_by_tbyx(results)

    if show:
        im = deepcopy(image)
        for i, cell in enumerate(results):
            x1,y1,x2,y2 = cell["box"]
            cv2.rectangle(
                im,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,0,0),2)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

    return results

def find_match(key_sentences, paddleOCR, box, eta=0.95): # Could be optimized
    """
    Detect if the key sentence is seen by the OCR.
    If it's the case return the index where the sentence can be found in the text returned by the OCR,
    else return an empty array
    Args:
        key_sentences (list) : contains a list with one are more sentences, each word is a string.
        text (list) : text part of the dict returned by pytesseract 

    Returns:
        res_indexes (list) : [[start_index, end_index], [empty], ...]   Contains for each key sentence of the landmark the starting and ending index in the detected text.
                            if the key sentence is not detected res is empty.
    """
    def _get_best(base_match, new_match):

        best = base_match
        if new_match == None:
            return best
        elif base_match.number_of_match < new_match.number_of_match: # Choose best match
            best = new_match
        elif (base_match.number_of_match == new_match.number_of_match): # If same number of match, choose the first one
            best=base_match
        return best
    
    xmin,ymin,xmax, ymax = box
    best_matches = None
    for i_place, dict_sequence in enumerate(paddleOCR):
        x1,y1 = dict_sequence["box"][:2]
        seq_match = None
        if xmin<x1<xmax and ymin<y1<ymax:
            sequence = dict_sequence["text"]
            for i_key, key in enumerate(key_sentences): # for landmark sentences from the json
                key_match = None
                for i_word, word in enumerate(sequence):
                    word = unidecode(word).lower()
                    for _, key_word in enumerate(key.split(" ")): # among all words of the landmark
                        key_word = unidecode(key_word).lower()
                        if word[:min(len(word),len(key_word))] == key_word:
                            distance = 1
                        else :
                            distance = jaro_distance("".join(key_word), "".join(word)) # compute the neighborood matching

                        if distance > eta : # take the matching neighborood among all matching words
                            if key_match == None:
                                key_match = KeyMatch(i_place, distance, 1, i_word, i_key, dict_sequence)
                            elif key_match.last_place_word<i_word:
                                key_match.confidence = min(key_match.confidence, distance)
                                key_match.number_of_match+=1
                if seq_match==None : 
                    seq_match=key_match
                else:
                    seq_match = _get_best(seq_match, key_match)

        if best_matches==None : 
            best_matches=seq_match
        else:
            best_matches = _get_best(best_matches, seq_match)
    
    # if best_matches != None : print(best_matches.OCR["text"], key_sentences[best_matches.key_index], best_matches.number_of_match)
    
    return best_matches

def clean_sequence(paddle_list, checkboxes, full="|\[]_!<>{}—;$€&*‘§—~+", left="'(*): |\[]_!.<>{}—;$€&-"): 
    res_dicts = []
    for dict_seq in paddle_list:

        if any([get_iou(dict_seq["box"], c["BOX"])>0.2 for c in checkboxes]):
            continue

        text = dict_seq["text"]

        text = re.sub(" :", ":", text)
        text = re.sub(":", ": ", text)
        text = re.sub("\(", " (", text)

        # Would increase "DATE B/L detection"
        text = re.sub("B/L", "B/L ", text)

        text = re.sub("`", "'", text)
        text = re.sub("_", " ", text)
        text = re.sub("_", " ", text)
        text = re.sub("I'", "l'", text)
        text = re.sub("-1", "", text)

        text = re.sub("  ", " ", text)

        if not text in full+left:
            text = [word.strip(full) for word in text.split(" ")]
            dict_seq["text"] = [word for word in text if word]
            res_dicts.append(dict_seq)

    return res_dicts

def get_key_matches_and_OCR(cropped_image, checkboxes, MODEL_HELPER=MODEL_HELPER, show=False):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        zone_match_dict (dict) :  { zone : Match,
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = cropped_image.shape[:2]
    zone_match_dict = {}

    # Search text on the whole image
    full_img_OCR =  paddle_OCR(cropped_image, show)
    full_img_OCR = clean_sequence(full_img_OCR, checkboxes)
    for zone, key_points in MODEL_HELPER.items():
        subregion = key_points["subregion"] # Area informations
        xmin, xmax = image_width*subregion["frac_x_min"], image_width*subregion["frac_x_max"]
        ymin, ymax = image_height*subregion["frac_y_min"], image_height*subregion["frac_y_max"]

        match = find_match(key_points["key_sentences"], full_img_OCR, (xmin,ymin,xmax, ymax)) if key_points["key_sentences"] else None
        # print("base : ", zone, (xmin, ymin, xmax, ymax))
        # plt.imshow(cropped_image[int(ymin):int(ymax), int(xmin):int(xmax)])
        # plt.show()

        if match:
            # print("found : ", zone, " - ", match.OCR["box"])
            zone_match_dict.update({zone : match })
            # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else :
           base_match = deepcopy(KeyMatch(0, -1, 0, 0, 0, NULL_OCR))
           base_match.OCR["box"] = [int(xmin), int(ymin), int(xmax), int(ymax)]
           zone_match_dict.update({zone : base_match})

    return zone_match_dict, full_img_OCR   

def get_area(cropped_image, box, relative_position, corr_ratio=1.1):
    """
    Get the area coordinates of the zone thanks to the landmark and the given relative position
    Args:
        box (list): detected landmark box [x1,y1,x2,y2]
        relative_position ([[vertical_min,vertical_max], [horizontal_min,horizontal_max]]): number of box height and width to go to search the tet
    """
    im_y, im_x = cropped_image.shape[:2]
    x1,y1,x2,y2 = box
    h, w = abs(y2-y1), abs(x2-x1)
    h_relative, w_relative = h*(relative_position["y_max"]-relative_position["y_min"])//2, w*(relative_position["x_max"]-relative_position["x_min"])//2
    y_mean, x_mean = y1+h*relative_position["y_min"]+h_relative, x1+w*relative_position["x_min"]+w_relative
    x_min, x_max = max(x_mean-w_relative*corr_ratio,0), min(x_mean+w_relative*corr_ratio*2, im_x)
    y_min, y_max = max(y_mean-h_relative*corr_ratio, 0), min(y_mean+h_relative*corr_ratio*2, im_y)
    (y_min, x_min) , (y_max, x_max) = np.array([[y_min, x_min], [y_max, x_max]]).astype(int)[:2]
    return x_min, y_min, x_max, y_max

def get_wanted_text(cropped_image, zone_key_match_dict, full_img_OCR, zone_key_dict, model, checkboxes=None, show=False):
    
    zone_matches = {}

    for zone, key_points in zone_key_dict.items():
        key_match =  zone_key_match_dict[zone]
        box = key_match.OCR["box"]
        condition, relative_position = key_points["conditions"], key_points["relative_position"]
        # If the key match is not found, the searching area is the subregion of the field
        xmin, ymin, xmax, ymax = box if key_match.confidence==-1 else get_area(cropped_image, box, relative_position, corr_ratio=1.15)
        
        candidate_dicts = [dict_sequence for dict_sequence in full_img_OCR if 
                      (xmin<dict_sequence["box"][0]<xmax) and (ymin<dict_sequence["box"][1]<ymax)]
                
        zone_match = ZoneMatch(candidate_dicts, [], 0, [])

        # print(zone, " : ", candidate_dicts)
   
        match_indices, res_seq = condition_filter(candidate_dicts, condition, model, application_path, ocr_pathes=OCR_PATHES, checkboxes=checkboxes)
        
        if len(res_seq)!=0:
            if type(res_seq[0]) != type([]):
                res_seq = unidecode(" ".join(res_seq))
        elif res_seq == []:
            res_seq =""

        zone_match.match_indices , zone_match.res_seq = match_indices, res_seq
        zone_match.confidence = min([candidate_dicts[i]["proba"] for i in zone_match.match_indices]) if zone_match.match_indices else 0

        zone_matches[zone] = {
                "sequence" : zone_match.res_seq,
                "confidence" : float(zone_match.confidence),
                "area" : (int(xmin), int(ymin), int(xmax), int(ymax))
            }
        
        if show:
            print(zone, ": ", res_seq)
            #print(box, key_match.confidence, (xmin, ymin, xmax, ymax))
            plt.imshow(cropped_image[ymin:ymax, xmin:xmax])
            plt.show()

    return zone_matches 

def model_particularities(zone_matches):
    # Add M/V to the navire
    zone_matches["navire"]["sequence"] = "M/V - " + zone_matches["navire"]["sequence"]

    # Delte scelles mistake
    zone_matches["scelles"]["sequence"] = "" if jaro_distance(zone_matches["scelles"]["sequence"].lower(), "autres")>0.90 else zone_matches["scelles"]["sequence"]

    # Select a product Code
    product_code_dict = ["Ble tendre", "Ble dur", "Fourrage orge", "Orge de brasserie"]

    marchandise = zone_matches["marchandise"]["sequence"]
    product_code = ""

    for pc in product_code_dict:
        if jaro_distance(unidecode(pc.upper()), marchandise)>0.95:
            product_code = pc
            
    zone_matches["code_produit"] = deepcopy(zone_matches["marchandise"])
    zone_matches["code_produit"]["sequence"] = product_code 

    return zone_matches

def textExtraction(cropped_image, zone_key_match_dict, full_img_OCR, model, checkboxes=None, zone_key_dict=MODEL_HELPER):
    """
    The main fonction to extract text from FDA

    Returns:
        zone_matches (dict) : { zone : {
                                    "sequence": ,
                                    "confidence": ,
                                    "area": }
        }
    """

    zone_matches = get_wanted_text(cropped_image, zone_key_match_dict, full_img_OCR, zone_key_dict, model, checkboxes=checkboxes, show=False)

    zone_matches = model_particularities(zone_matches)

    for zone, dict in zone_matches.items():
        print(zone, ":", dict["sequence"])

    return zone_matches

def main(scan_dict, model=MODEL):

    pdfs_res_dict = {}

    for pdf, images_dict in scan_dict.items():
        print("###### Traitement de :", pdf, " ######")
        pdfs_res_dict[pdf] = {}
        for i_image, (image_name, sample_image) in enumerate(list(images_dict.items())):
            print("--------------", image_name, "--------------")

            image = binarized_image(sample_image)  
            # image = get_adjusted_image(image, show=False)
            # plt.imsave("im.png", image, cmap="gray")

            templates_pathes = [os.path.join(CHECK_PATH, dir) for dir in os.listdir(CHECK_PATH) if os.path.splitext(dir)[1].lower() in [".png", ".jpg"]]
            checkboxes = get_checkboxes(image, templates_pathes=templates_pathes, show=False) # List of checkbox dict {"TOP_LEFT_X"...}

            # plt.imshow(image)
            # plt.show()

            zone_key_match_dict, full_img_OCR = get_key_matches_and_OCR(image, checkboxes, show=False)

            sample_matches = textExtraction(image, zone_key_match_dict, full_img_OCR, model, checkboxes=checkboxes, zone_key_dict=MODEL_HELPER)

            # Here one scan = one sample
            pdfs_res_dict[pdf][f"sample_{i_image}"] = {"IMAGE" : image_name,
                                              "EXTRACTION" : sample_matches} # Image Name
    
    return pdfs_res_dict

if __name__ == "__main__":

    path = r"C:\Users\CF6P\Desktop\ECAR\Data\CU\NON OAIC\test\NOAIC_test_xl.pdf"

    from ProcessPDF import PDF_to_images
    import os

    images = PDF_to_images(path)

    images = images[:]
    images_names = ["res"+f"_{i}" for i in range(1,len(images)+1)]


    scan_dict = {"debug" : {}}
    for im_n, im in zip(images_names, images):
        scan_dict["debug"].update({im_n : im})

    main(scan_dict, model="CU OAIC")
import numpy as np
import pandas as pd
 
import re
import os

from copy import deepcopy
from JaroDistance import jaro_distance
from unidecode import unidecode

from datetime import datetime
year = datetime.now().year

def _get_sheet(path, condition_dict, model):

    if not "sheet_name" in condition_dict.keys():
        df = pd.read_excel(path)
    
    else:
        sheet_name = condition_dict["sheet_name"]

        xl = pd.ExcelFile(path)
        sheet_names = xl.sheet_names  # see all sheet names

        if sheet_name in sheet_names:
            df = xl.parse(sheet_name)
        elif sheet_name+" "+model in sheet_names:
            df = xl.parse(sheet_name+" "+model)
        else :
            df = pd.read_excel(path)

    return df

def list_process(candidates, condition_dict, lists_df, min_jaro = 0.87):
    """ Search candidates that contains all the word from a list of word/sequence, with a word by word 'min_jaro' tolerance"

    Args:
        candidates (list of dict): 
        condition (list of three elements): ["list", column of lists_df to extract all the target words, multiple or single]
        lists_df (DataFrame): _description_
    """

    def sub_mistakes(word):
        # Replace common mistakes from the OCR to find a perfect match
        word = word.replace("O", "0") if "O" in word else word
        word = word.replace("l", "I") if "l" in word else word
        return word

    def best_matches_dict(check_word, candidate_sequence, candidate_index):
        """ 
        Returned a dict wich carry the information about the best matched word among all sequences
        """
        max_jaro = min_jaro
        status_dict = {
            "find" : False,
            "index" : -1,
            "jaro" : 0,
            "word" : ""
        }

        for i_word, word in enumerate(candidate_sequence):
            if unidecode(check_word.lower()) in [unidecode(word.strip(": _;").lower()), unidecode(sub_mistakes(word).strip(": _;").lower())]:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = 1, check_word
                return status_dict
            
            jaro = jaro_distance(unidecode(word.lower()), unidecode(check_word.lower()))
            if jaro > max_jaro:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = jaro, check_word

        return status_dict

    all_text, all_indices = [], []
    matched_elmts = []
    mode = condition_dict["mode"]

    for i_text, dict_text in enumerate(candidates):
        all_text+=dict_text["text"]
        all_indices+= [i_text for i in range(len(dict_text["text"]))]
    
    check_list = list(lists_df[condition_dict["column"]].dropna().unique())

    for check_elmt in check_list:
        found_elmts = []
        check_words = check_elmt.split(" ")
        for check_word in check_words:
            found_dict = best_matches_dict(check_word, all_text, all_indices)
            if found_dict["find"]: # If a check word is found : stack it
                found_elmts.append(found_dict)
    
            if 0<len(found_elmts)<=len(check_words) and check_elmt not in [matched_elmt["element"] for matched_elmt in matched_elmts]: # if some checking ellement are found and there are not assigned yet
                
                matched_elmts.append({
                                    "element": check_elmt,
                                    "words" : found_elmts,
                                    "jaro" : min([d["jaro"] for d in found_elmts]),
                                    "index" : np.median(np.array([d["index"] for d in found_elmts], dtype=np.int64))
                                    })
                
    matched_elmts = sorted(matched_elmts, key=lambda x: x["index"])

    res_seq, match_indices = [], []
    if matched_elmts:
        if mode == "multiple": # Return all found elements sorted by index
            res_seq, match_indices = [[matched_elmt["element"]] for matched_elmt in matched_elmts], [int(matched_elmt["index"]) for matched_elmt in matched_elmts]
        if mode == "single":
            matched_elmts = sorted(matched_elmts, key=lambda x: (-len(x["words"]), -x["jaro"], x["index"]), reverse=False)
            res_seq, match_indices = [matched_elmts[0]["element"]], [int(matched_elmts[0]["index"])]

    return match_indices, res_seq

def cell_process(candidates):
    match_indices, res_seq = [], []
    for i_text, dict_text in enumerate(candidates):
        res_seq+=dict_text["text"]
        match_indices+=[i_text]
    return match_indices, res_seq

def int_process(candidates):
    match_indices, res_seq = [], []
    for i_text, dict_text in enumerate(candidates):
        if dict_text["text"][0].isdigit():
            res_seq=[dict_text["text"][0]]
            match_indices+=[i_text]

    return match_indices, res_seq

def after_key_process(candidates, bound_keys, similarity=0.85):

    def _get_wanted_seq(full_seq, target_words, search_range):
        """
        Find the place of the target_word and the following of the sentence after the word
        
        Return the index and new full sequence
        """
        res_index = 0
        for target_word in target_words:
            target_word = unidecode(target_word).lower()
            for place in range(min(search_range+2, len(full_seq))):
                word = unidecode(full_seq[place]).lower()
                try:
                    index = word.rindex(target_word)
                    res_word = word[index+len(target_word):]
                    if res_word == "" and place < len(full_seq)-1: # The target is at the end of a word
                        res_index+=place
                        full_seq = full_seq[place+1:]
                        break
                    if res_word == "" and place == len(full_seq)-1: # The target is at the end of the seq
                        return -1, []
                    
                    full_seq[place] = res_word
                    res_index+=place
                    full_seq = full_seq[place:]
                    break
                
                except ValueError:
                    pass
        
        return res_index, full_seq
    
    strip = ' ().*:‘;,§"'+"'"
    key_boundaries = {"after" : [], "before" : []} 

    # Get the all matched between keys and sequences for the start and the end key
    for state, bound_key in bound_keys.items():
        if bound_key ==  "":
            continue
        bound_word = bound_key.split(" ")
        for i_key, key_word in enumerate(bound_word):
            for i_candidate, candidate_sequence in enumerate(candidates):
                    for i_word, word in enumerate(candidate_sequence["text"]):
                        word, key_word = unidecode(word).lower(), unidecode(key_word).lower()
                        word_match = {"i_key" : -1,
                                      "i_candidate" : -1,
                                      "i_word" : -1} # Store the matching key index and the place on the candidate seq
                        # If the rwo words are close
                        if (jaro_distance(key_word, unidecode(word).strip(strip))>similarity):
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[state].append(word_match)
                            break
                        # If a sequence word is containing a key_word
                        if len(key_word)>2 and key_word in word:
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[state].append(word_match)
                            break
    
    if key_boundaries["after"] == [] :
        if len(candidates)==1:
            print("DEFAULT CASE :", candidates[0]["text"])
            return [0], candidates[0]["text"]
        return [], []

    # Get boundaries of the desired text
    _order_by_match_candidate = lambda match_list : sorted(match_list, 
                                                key=lambda match : ([d["i_candidate"] for d in match_list].count(match["i_candidate"]), -match["i_candidate"]), 
                                                                    reverse=True) # Most matched sentence, the firt one in case of tie
   
    last_start_match = sorted(_order_by_match_candidate(key_boundaries["after"]), key=lambda d:d["i_key"], reverse=True)[0]

    if last_start_match["i_word"] == len(candidates[last_start_match["i_candidate"]]["text"])-1:
        if last_start_match["i_candidate"] == len(candidates)-1:
            return [], []
        last_start_match["i_candidate"], last_start_match["i_word"] = last_start_match["i_candidate"]+1, 0

    if key_boundaries["before"]==[]:
        first_end_seq_id = deepcopy(last_start_match) # Equivalent to a mis match : Will be changed in the next step
    else:
        first_end_seq_id = sorted(_order_by_match_candidate(key_boundaries["before"]), key=lambda d:d["i_key"], reverse=False)[0]
        # Empty field case
        if (first_end_seq_id["i_candidate"], first_end_seq_id["i_word"]) == (last_start_match["i_candidate"], last_start_match["i_word"]):
            return [],  []
 
    # If there is a mismatch, end is set as the sequence that followed the start
    if first_end_seq_id["i_candidate"] <= last_start_match["i_candidate"] :
        first_end_seq_id["i_candidate"] = min(last_start_match["i_candidate"]+1, len(candidates))
        first_end_seq_id["i_word"] = 0

    text_candidates = [candidates[i]["text"] for i in range(last_start_match["i_candidate"], first_end_seq_id["i_candidate"])]
    text_candidates[0] = text_candidates[0][last_start_match["i_word"]:]

    # Get all the found text as a list of string words
    all_text = [word for text_candidate in text_candidates for word in text_candidate]
    all_local_indices = [i for i,text in enumerate(text_candidates) for s in range(len(text))]

    search_range = (len(bound_keys["after"]) - last_start_match["i_key"]) # 0> if the last detected word is not the last key word 
    target_words = [bound_keys["after"].split(" ")[-1]] + ["(*):"]
    
    line_index, res_seq = _get_wanted_seq(all_text, target_words, search_range)

    return [all_local_indices[line_index]], res_seq

def date_process(candidates, strip):
    match_indices, res_seq = [], []
    for i_candidate, candidate in enumerate(candidates):
        try:
            word_test = " ".join(candidate["text"])
            date  = datetime.strptime(word_test, "%B %d, %Y")
            return [i_candidate], [date.strftime("%d/%m/%Y")]
        except ValueError:
            pass
    
        for i_word, word in enumerate(candidate["text"]):
            try:
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try:
                word_test = word[:10].lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass
            
            try: # Case dd/mm/yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                word_test = word_test[:-2] + "20" + word_test[-2:]
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try: # Case dd,mm,yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                word_test = word_test[:-2] + "20" + word_test[-2:]
                _ = bool(datetime.strptime(word_test, "%d,%m,%Y"))
                return [i_candidate], [word_test.replace(",", "/")]
            except ValueError:
                pass

            try: # Case dd-mm-yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                _ = bool(datetime.strptime(word_test, "%d-%m-%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try: # Case month in letter
                word = word.lower().strip(strip)
                _ = bool(datetime.strptime(word, "%B"))
                full_date = "".join(candidate[i_word-1:i_word+2])
                _ = bool(datetime.strptime(full_date, "%d%B%Y"))
                return [i_candidate], [word]
            except:
                continue
            
    return match_indices, res_seq

def contain_process(candidates, keys):
    matched_candidates = []
    matched_indices = []
    for i_cand, candidate in enumerate(candidates):
        for i_word, word in enumerate(candidate["text"]):
            for _, key_word in enumerate(keys):
                if key_word in word:
                    matched_candidates.append(word)
                    matched_indices.append(i_cand)

    return matched_indices, matched_candidates

def format_process(candidates, keys):
    
    def _format_start_with(key_word, word):
        if key_word == "YYMM":
            YY = year-2000
            if word[:3].isnumeric() and len(word)>3:
                if YY-2<int(word[:2])<YY+1 and 0<int(word[2:4])<13:
                    return True
        
        if set(key_word) == {"N"}:
            if word[:len(key_word)].isnumeric() and len(word)>=len(key_word):
                return True

        elif word.startswith(key_word):
            return True

        return False

    best_count_index, best_count = -1, 0
    for i_cand, candidate in enumerate(candidates):
        count = 0
        word = "".join(candidate["text"])
        word = re.sub("_-/\[]]", "", word)
        for _, key_word in enumerate(keys):
            key_word = re.sub("_-/\[]]", "", key_word)
            if _format_start_with(key_word, word):
                count+=1
                word = word[len(key_word):]
        if count>best_count:
            best_count=count
            best_count_index = i_cand       

    if best_count_index == -1:
        return [], []

    matched_indices, matched_candidates = [best_count_index], candidates[best_count_index]["text"]

    return matched_indices, matched_candidates

def check_process(candidates, sense, checkboxes, lists_df, list_condition_dict):
    """_summary_

    Args:
        candidates (list of dict):  of candidate dict
        sense (right or left): Direction where to search after the checkbox as been detected
        checkboxes (list of arrays): All screened checkboxes
        lists_df (pd.Dataframe): df of all lists as columns
        list_condition_dict (dict): "mode" : single or multiple values ; "column" : the column where of the list

    Returns:
        candidates_indices, check_candidates
    """
    
    check_candidates = []
    candidates_indices = []

    # First extract candidates according to checkboxes
    for checkbox in checkboxes:
        min_dist = 10000
        nearest_candidate, n_index = None, None

        up, down = checkbox["TOP_LEFT_Y"], checkbox["BOTTOM_RIGHT_Y"]
        left, right = checkbox["TOP_LEFT_X"], checkbox["BOTTOM_RIGHT_X"]

        for i_cand, candidate in enumerate(candidates):
            x1,y1,x2,y2 = candidate["box"]
            # If a checkbox and a word are in the same height
            if (up<=y1<=down) or (up<=y2<=down) or (y1<=up<=y2) or (y1<=down<=y2):
                dist = left-x2 if sense == "left" else x1-right
                if 0<dist<min_dist:
                    min_dist = dist
                    nearest_candidate, n_index = candidate, i_cand

        # Found list
        if nearest_candidate and list_condition_dict:

            _, res_seq = list_process([nearest_candidate], list_condition_dict, lists_df)
            
            if res_seq and not res_seq in check_candidates:
                check_candidates.append(res_seq)
                candidates_indices.append(i_cand)
    
    return candidates_indices, check_candidates

def Nechantillon_ELPV(candidates):
    match_indices, res_seq = [], []
    base1, base2 = year-1, year+2 # Thresholds
    
    for i_candidate, candidate in enumerate(candidates):
        if len(candidate["text"])<4:
            for i_word, word in enumerate(candidate["text"]):
                if len(word)>3 and not "/" in word:
                    try_list = [(base1-2000, base2-2000), (base1,base2)]
                    for date_tuple in try_list:
                        num1, num2 = date_tuple
                        if word[:len(str(num1))].isnumeric():
                            date_num, code_num = word[:len(str(num1))], word[len(str(num1)):].upper()
                            if num1 <= int(date_num) < num2 : # Try to avoid strings shorter than NUM
                                res="".join(candidate["text"][i_word:])
                                res_upper = res.upper()
                                
                                # Replace common mistake
                                correction_list = [("MPA", "MP4"), ("N0", "NO"), ("AUOP", "AU0P"), ("CEOP", "CE0P"), ("GEL0", "GELO"), ("PLOP", "PL0P"), 
                                    ("PLIP", "PL1P"), ("NCIP", "NC1P"), ("NCIE", "NC1E"), ("S0R", "SOR"), ("1F", "IF")]
                                for cor_tuple in correction_list:
                                    error, correction = cor_tuple
                                    if code_num[:len(error)] == error:
                                        res_upper = res_upper.replace(error, correction, 1)
                                        break # One possibility
                                if code_num[5:9] == "S0PDT": # Unique case
                                    res_upper.replace("SP0DT", "SOPODT", 1)
                                
                                match_indices, res_seq = [i_candidate], [res_upper]
                    # Special case
                    if "GECA" in word:
                        res_upper = str(year)+word if str(year) not in word else word
                        try :
                            if candidate["text"][i_word+1].isnumeric():
                                res_upper += candidate["text"][i_word+1]
                        except:
                            pass
                        match_indices, res_seq = [i_candidate], [res_upper]

    return match_indices, res_seq

def condition_filter(candidates_dicts, condition, model, application_path, ocr_pathes, checkboxes=None):
    """_summary_

    Args:
        candidates_dicts (_type_): _description_
        key_main_sentences (_type_): _description_
        conditions (_type_): _description_

    Returns:
        _type_: _description_
    """
    strip =  "|\[]_!<>{}—;$€&*‘§—~-'(*): " + '"'
    # Arbitrary set
    candidates = deepcopy(candidates_dicts)
    match_indices, res_seq = [], []

    if condition[0] == "contains":
        keys = condition[1]
        match_indices, res_seq = contain_process(candidates, keys)

    if condition[0] == "after_key":
        bound_keys = condition[1] # Start and end key_sentences
        match_indices, res_seq = after_key_process(candidates, bound_keys)
            
    if condition[0] == "date": # Select a date format
        match_indices, res_seq = date_process(candidates, strip)

    if condition[0] == "list": # In this case itertion is over element in the condition list
        condition_dict = condition[1]
        path = os.path.join(application_path, ocr_pathes[condition_dict["path"]])

        lists_df = _get_sheet(path, condition_dict, model)
        
        match_indices, res_seq= list_process(candidates, condition_dict, lists_df)

    if condition[0] == "format": # In this case itertion is over element in the condition list
        keys = condition[1]
        match_indices, res_seq = format_process(candidates, keys)
    
    if condition[0] == "constant":
        match_indices, res_seq = [], [condition[1]]
    
    if condition[0] == "checkbox":
        # The sens where to extract sentences starting from the checkboxe
        sense = condition[1]
        # If a list could help to find the sentence
        try:
            list_condition_dict = condition[2]
            # if the list file is global        
            path = os.path.join(application_path, ocr_pathes[list_condition_dict["path"]])
            lists_df = _get_sheet(path, list_condition_dict, model)
        except:
            list_condition_dict, lists_df = None, None

        match_indices, res_seq = check_process(candidates, sense, checkboxes, lists_df, list_condition_dict)
    
    if condition[0] == "echantillon_ELPV": # A special filter for numero d'echantillon
        match_indices, res_seq = Nechantillon_ELPV(candidates)
    
    if condition[0] == "cell": # A special filter for numero d'echantillon
        match_indices, res_seq = cell_process(candidates)

    if condition[0] == "int": # A special filter for numero d'echantillon
        match_indices, res_seq = int_process(candidates)

    return match_indices, res_seq

if __name__ == "__main__":

    import sys
    import json

    candidates = [{'text': ['QUALITE', 'DU', 'GRAIN'], 'box': [159.0, 738.0, 775.0, 804.0], 'proba': 0.999}, {'text': ['..'], 'box': [1032.0, 754.0, 1212.0, 795.0], 'proba': 0.826}, {'text': ['Poids', 'Spécifique', '(QF015)'], 'box': [149.0, 840.0, 637.0, 907.0], 'proba': 0.998}, {'text': ['0'], 'box': [871.0, 840.0, 945.0, 903.0], 'proba': 0.989}, {'text': ['Humidité', 'uniquement', '(AA07A)'], 'box': [149.0, 906.0, 726.0, 987.0], 'proba': 0.983}, {'text': ['0'], 'box': [867.0, 914.0, 945.0, 979.0], 'proba': 0.993}, {'text': ['Protéines', 'sur', 'sec', 'x5,7', '(incluant', 'humidité)'], 'box': [151.0, 994.0, 989.0, 1049.0], 'proba': 0.985}, {'text': ['Dumas', '(QF008+QF0039)'], 'box': [987.0, 986.0, 1463.0, 1068.0], 'proba': 0.962}, {'text': ['/Kjeldhal', '(20483)'], 'box': [1565.0, 998.0, 1938.0, 1049.0], 'proba': 0.989}, {'text': ['0'], 'box': [2137.0, 983.0, 2188.0, 1030.0], 'proba': 0.768}, {'text': ['Hagberg', '(M0270)'], 'box': [150.0, 1062.0, 466.0, 1145.0], 'proba': 0.989}, {'text': ['Zélény', '(QF003)'], 'box': [1015.0, 1067.0, 1410.0, 1140.0], 'proba': 0.819}, {'text': ['(QF016+AA07A+QF037)'], 'box': [1768.0, 1056.0, 2111.0, 1103.0], 'proba': 0.999}, {'text': ['Alvéogramme', '(QF001', 'A7500)'], 'box': [152.0, 1147.0, 693.0, 1217.0], 'proba': 0.967}, {'text': ['Gluten', 'Index', '(QF01P)'], 'box': [1312.0, 1139.0, 1718.0, 1218.0], 'proba': 0.992}, {'text': ['0'], 'box': [1864.0, 1136.0, 1934.0, 1199.0], 'proba': 0.993}, {'text': ['Gluten', 'humide'], 'box': [159.0, 1228.0, 461.0, 1275.0], 'proba': 0.993}, {'text': ['-', 'mécanique', '(21415-2)', '(QF0AZ)'], 'box': [466.0, 1216.0, 1114.0, 1290.0], 'proba': 0.96}, {'text': ['/', 'manuel', '(21415)', '(QF004)'], 'box': [1197.0, 1213.0, 1818.0, 1290.0], 'proba': 0.968}, {'text': ['0'], 'box': [1867.0, 1224.0, 1926.0, 1272.0], 'proba': 0.924}, {'text': ['Impuretés'], 'box': [154.0, 1309.0, 381.0, 1351.0], 'proba': 0.996}, {'text': ['ISO15587', '(QF04M)'], 'box': [538.0, 1289.0, 868.0, 1372.0], 'proba': 0.994}, {'text': ['/', 'ISO7970', '(QF04R)'], 'box': [1008.0, 1293.0, 1427.0, 1367.0], 'proba': 0.922}, {'text': ['0'], 'box': [1613.0, 1294.0, 1679.0, 1356.0], 'proba': 0.988}, {'text': ['-', 'si', 'demande', 'spécifiques,', 'préciser:'], 'box': [218.0, 1385.0, 812.0, 1432.0], 'proba': 0.988}, {'text': ['Insectes'], 'box': [155.0, 1458.0, 343.0, 1509.0], 'proba': 0.996}, {'text': ['Vivants', '(QF045)'], 'box': [530.0, 1450.0, 813.0, 1521.0], 'proba': 0.969}, {'text': ['0'], 'box': [971.0, 1451.0, 1044.0, 1509.0], 'proba': 0.989}, {'text': ['1'], 'box': [1099.0, 1446.0, 1134.0, 1509.0], 'proba': 0.989}, {'text': ['Morts', '(QF044)'], 'box': [1250.0, 1450.0, 1492.0, 1521.0], 'proba': 0.895}, {'text': ['0'], 'box': [1694.0, 1458.0, 1749.0, 1513.0], 'proba': 0.973}, {'text': ['Prédateurs', 'vivants', '(arost)-', 'Trogoderma'], 'box': [152.0, 1527.0, 967.0, 1593.0], 'proba': 0.962}, {'text': ['Prostephanus'], 'box': [1053.0, 1527.0, 1375.0, 1590.0], 'proba': 0.999}, {'text': ['Anguina'], 'box': [1504.0, 1527.0, 1707.0, 1590.0], 'proba': 0.999}, {'text': ['/', 'yraus'], 'box': [1790.0, 1531.0, 1993.0, 1582.0], 'proba': 0.928}, {'text': ['Graines', 'toxiques', 'et', 'nuisibles', '(QFo5P)'], 'box': [156.0, 1597.0, 907.0, 1670.0], 'proba': 0.979}, {'text': ['0'], 'box': [1044.0, 1604.0, 1100.0, 1652.0], 'proba': 0.849}, {'text': ['/', 'Graines', 'Toxiques', 'Ergot', '(QF046)'], 'box': [1311.0, 1600.0, 1985.0, 1670.0], 'proba': 0.969}, {'text': ['ANALYSES', 'SANITAIRES'], 'box': [166.0, 1710.0, 919.0, 1776.0], 'proba': 1.0}, {'text': ['Aflatoxines', 'Immuno', '-'], 'box': [162.0, 1798.0, 664.0, 1849.0], 'proba': 0.96}, {'text': ['B1+B2+G1+G2', '(QF0AD)'], 'box': [836.0, 1782.0, 1275.0, 1861.0], 'proba': 0.987}, {'text': ['1', 'B1', '(QF0AE)'], 'box': [1425.0, 1771.0, 1734.0, 1865.0], 'proba': 0.893}, {'text': ['Ochratoxines', 'A', 'Immuno', '(QF01G)'], 'box': [160.0, 1859.0, 792.0, 1937.0], 'proba': 0.981}, {'text': ['0'], 'box': [930.0, 1856.0, 1004.0, 1918.0], 'proba': 0.995}, {'text': ['1'], 'box': [1089.0, 1878.0, 1118.0, 1915.0], 'proba': 0.529}, {'text': ['DON', '(Vomitoxines)', 'Immuno', '(QF01H)'], 'box': [1159.0, 1863.0, 1878.0, 1933.0], 'proba': 0.972}, {'text': ['0.'], 
    'box': [1956.0, 1856.0, 2033.0, 1918.0], 'proba': 0.978}, {'text': ['Zéaralénone', 'Immuno', '(QF011)'], 'box': [156.0, 1936.0, 719.0, 2010.0], 'proba': 0.974}, 
    {'text': ['Mycotoxines', 'par', 'HPLC.', '(lesquelles):'], 'box': [162.0, 2028.0, 908.0, 2079.0], 'proba': 0.979}, {'text': ['Radioactivité', '(RR00L+RR0oM+RRGAD)'], 'box': [156.0, 2108.0, 807.0, 2193.0], 'proba': 0.948}, {'text': ['Pesticides', '(pack)', '(PZVPA', 'ZVR08', 'QF0AY)'], 'box': [156.0, 2207.0, 896.0, 2288.0], 'proba': 0.978}, {'text': ['CODEX', 'Alimentarius'], 'box': [1118.0, 2218.0, 1506.0, 2266.0], 'proba': 0.992}, {'text': ['/"', 'Autres'], 'box': [1753.0, 2222.0, 1938.0, 2273.0], 'proba': 0.931}, {'text': ['0'], 'box': [2000.0, 2211.0, 2070.0, 2273.0], 'proba': 0.987}, {'text': ['Phosphine', '(SF0os', 'SFXTR)'], 'box': [164.0, 2287.0, 644.0, 2361.0], 'proba': 0.96}, {'text': ['0'], 'box': [1022.0, 2291.0, 1089.0, 2357.0], 'proba': 0.918}, {'text': ['/', 'Glyphosate', '(SFo0', 'SFXTR)'], 'box': [1314.0, 2287.0, 1859.0, 2357.0], 'proba': 0.875}, {'text': ['0'], 'box': [1982.0, 2284.0, 2056.0, 2350.0], 'proba': 0.984}, {'text': ['Pirimiphos', 'méthyl', '(AA20l', 'AAZ00', 'QF0AY)'], 'box': [160.0, 2360.0, 907.0, 2442.0], 'proba': 0.942}, {'text': ['0'], 'box': [1022.0, 2368.0, 1089.0, 2430.0], 'proba': 0.995}, {'text': ['1', '1,2-Dibromethane', '(AABHs)'], 'box': [1319.0, 2356.0, 1874.0, 2434.0], 'proba': 0.975}, {'text': ['0'], 'box': [1985.0, 2361.0, 
    2056.0, 2426.0], 'proba': 0.983}, {'text': ['Deltaméthrin', '(AA23C', 'AAZ00', 'QF0AY)'], 'box': [164.0, 2433.0, 818.0, 2511.0], 'proba': 0.981}, {'text': ['0'], 'box': [1022.0, 2441.0, 1089.0, 2503.0], 'proba': 0.986}, {'text': ['/', 'Cyperméthrin', '(AA239', 'AAZ00', 'QF0AY)'], 'box': [1318.0, 2433.0, 2029.0, 2511.0], 
    'proba': 0.933}, {'text': ['0'], 'box': [2133.0, 2448.0, 2203.0, 2510.0], 'proba': 0.979}, {'text': ['4', 'métaux', '(', 'Pb,', 'Cd,', 'Hg,', 'et', 'As)', '(Pzw6', 'ZvR08)'], 'box': [159.0, 2517.0, 1018.0, 2587.0], 'proba': 0.919}, {'text': ['0'], 'box': [1174.0, 2521.0, 1244.0, 2583.0], 'proba': 0.99}, {'text': ['2', 'métaux', '(Pb,', 'Cd)', '(zvWP0+ZW20+ZWW07+ZVR08)'], 'box': [160.0, 2583.0, 1014.0, 2664.0], 'proba': 0.935}, {'text': ['0'], 'box': [1174.0, 2594.0, 1247.0, 2657.0], 'proba': 0.993}, {'text': ['LevureS', '(L0001', 'KM0EK)'], 'box': [168.0, 2696.0, 589.0, 2767.0], 'proba': 0.878}, {'text': ['0'], 'box': [745.0, 2697.0, 819.0, 2759.0], 'proba': 0.991}, {'text': ['Moisisures', '(L0001', 'KMOEL).'], 'box': [1396.0, 2696.0, 1885.0, 2763.0], 'proba': 0.953}, {'text': ['0'], 'box': [1974.0, 2704.0, 2026.0, 2755.0], 'proba': 0.747}, {'text': ['Cyanures', '(QF07P)'], 'box': [161.0, 2787.0, 481.0, 2866.0], 'proba': 0.96}, {'text': ['0'], 'box': [627.0, 2792.0, 686.0, 2839.0], 'proba': 0.58}, {'text': ['Tilletia', 'indica', 'et', 'controversa', '(aro73)'], 'box': [1037.0, 2795.0, 1705.0, 2854.0], 'proba': 0.989}, {'text': ['0'], 'box': [1864.0, 2792.0, 1919.0, 2847.0], 'proba': 0.991}, {'text': ['OGM', '(PAX02)'], 'box': [166.0, 2860.0, 388.0, 2936.0], 'proba': 0.968}, {'text': ['0'], 'box': [635.0, 2865.0, 690.0, 2916.0], 'proba': 0.958}, {'text': ['Remarques', 'ou', 'autres', 'demandes'], 'box': [166.0, 2949.0, 760.0, 
    2985.0], 'proba': 0.969}, {'text': ['0000-0884E'], 'box': [416.0, 3070.0, 769.0, 3117.0], 'proba': 0.982}, {'text': ['0000-0885EX/20'], 'box': [944.0, 3041.0, 1392.0, 3102.0], 'proba': 0.94}, {'text': ['V', 'Soft'], 'box': [620.0, 3135.0, 720.0, 3161.0], 'proba': 0.76}, {'text': ['31é', 't'], 'box': [960.0, 3117.0, 1037.0, 3143.0], 'proba': 0.605}, {'text': ['endre', '/', 'Soft'], 'box': [1037.0, 3113.0, 1262.0, 3146.0], 'proba': 0.954}, {'text': ['Bié', 'tendre'], 'box': [428.0, 3146.0, 609.0, 3172.0], 'proba': 0.872}, {'text': ['4CLi:', 'M7V', 'ARKL.0W', 'MUSE'], 'box': [960.0, 3150.0, 1303.0, 3175.0], 'proba': 0.893}, {'text': ['Fn:'], 'box': [1520.0, 3340.0, 1605.0, 3365.0], 'proba': 0.7}]
    
    condition = ["checkbox", "left",  {"path":"contract_analysis_path", "sheet_name" : "analyse", "column" :"Denomination", "mode":"single"}]
    
    checkboxes = [{'BOX': [1343, 641, 1375, 670], 'TOP_LEFT_X': 1343, 'TOP_LEFT_Y': 641, 'BOTTOM_RIGHT_X': 1375, 'BOTTOM_RIGHT_Y': 670, 'MATCH_VALUE': 0.8551127, 'TEMPLATE': 'check11.png', 'COLOR': (255, 0, 0)}, {'BOX': [1594, 1013, 1628, 1045], 'TOP_LEFT_X': 1594, 'TOP_LEFT_Y': 1013, 'BOTTOM_RIGHT_X': 1628, 'BOTTOM_RIGHT_Y': 1045, 'MATCH_VALUE': 0.8382654, 'TEMPLATE': 'check14.png', 'COLOR': (255, 0, 0)}, {'BOX': [549, 1079, 583, 1114], 'TOP_LEFT_X': 549, 'TOP_LEFT_Y': 1079, 'BOTTOM_RIGHT_X': 583, 'BOTTOM_RIGHT_Y': 1114, 'MATCH_VALUE': 0.8858176, 'TEMPLATE': 'check12.png', 'COLOR': (255, 0, 0)}, {'BOX': [883, 1154, 917, 1186], 'TOP_LEFT_X': 883, 'TOP_LEFT_Y': 1154, 'BOTTOM_RIGHT_X': 917, 'BOTTOM_RIGHT_Y': 1186, 'MATCH_VALUE': 0.8639079, 'TEMPLATE': 'check14.png', 'COLOR': (255, 0, 0)}, {'BOX': [1435, 1800, 1469, 1832], 'TOP_LEFT_X': 1435, 'TOP_LEFT_Y': 1800, 'BOTTOM_RIGHT_X': 1469, 'BOTTOM_RIGHT_Y': 1832, 'MATCH_VALUE': 0.8784184, 'TEMPLATE': 'check14.png', 'COLOR': (255, 0, 0)}, {'BOX': [978, 2128, 1022, 2161], 'TOP_LEFT_X': 978, 'TOP_LEFT_Y': 2128, 'BOTTOM_RIGHT_X': 1022, 'BOTTOM_RIGHT_Y': 2161, 'MATCH_VALUE': 0.87540835, 'TEMPLATE': 'check16.png', 'COLOR': (255, 0, 0)}, {'BOX': [1629, 2229, 1663, 2264], 'TOP_LEFT_X': 1629, 'TOP_LEFT_Y': 2229, 'BOTTOM_RIGHT_X': 1663, 'BOTTOM_RIGHT_Y': 2264, 'MATCH_VALUE': 0.80177027, 'TEMPLATE': 'check12.png', 'COLOR': (255, 0, 0)}]

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

    print(condition_filter(candidates, condition, model="CU hors OAIC", application_path=application_path, ocr_pathes=OCR_PATHES, checkboxes=checkboxes))
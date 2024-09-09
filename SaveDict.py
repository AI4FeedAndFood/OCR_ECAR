import os
import dicttoxml
import json

from shutil import copyfile
from copy import deepcopy
from datetime import datetime
from PIL import Image

def _update_dict(stack_dict, value, keys_path):
    keys_path = keys_path.split(".") if type(keys_path)==type("") else keys_path
    if len(keys_path) == 0:
        return 
    if len(keys_path) == 1:
        stack_dict[keys_path[0]] = value
        return 
    key, *new_keys = keys_path
    if key not in stack_dict:
        stack_dict[key] = {}
    _update_dict(stack_dict[key], value, new_keys)
    return

def _get_dict_value(dict, keys_path):

    value = dict
    
    for key in keys_path.split("."):

        try:
            value = value[key]
        
        # If the key not in the dict
        except KeyError:
            return False
        
    return value

def runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows):

    analyses, comments = [], []

    for key, items in verif_values.items():
        # Analysis case
        if key[0] == "ana":
            index = key[1]
            if not index in deleted_rows:
                analyses.append(items)
                # comments.append(verif_values[("spec", index)])

            elif "spec" in key:
                continue
        # Client case
        elif key[0] == "client":
            if items:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[0]] = key[1]

        # Other cases
        elif key[0] == "zone":
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[1]]["sequence"] = items


    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse"]["sequence"] = analyses
    # res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse_specification"] = {"sequence" : comments}

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
    
    return res_dict

def keepNeededFields(verified_dict, client_contract, model, productcode_dict):
    """Extract fields to return to the lims from the interface

    Args:
        verified_dict (_type_): _description_
        client_contract (_type_): _description_
        model (_type_): _description_
    """

    added_field = ["analyse_specification", "client", "code_produit"]

    # dict for clean fields
    sample_clean_dict = {}
    for key, value  in verified_dict.items():
        if not key in added_field:
            sample_clean_dict[key] = value["sequence"]

    # Add the code_produit
    if "code_produit" in verified_dict.keys():
        sample_clean_dict["code_produit"] = productcode_dict[verified_dict["code_produit"]["sequence"]] if verified_dict["code_produit"]["sequence"] else ""

    # Find the Contract and the quotation according to the data
    client = verified_dict["client"]
    if "CU" in model:
        clientName = f"CU_{client}"
        oaic = "Oui" if model == "CU OAIC" else "Non"
        corresponding_row = client_contract[(client_contract["ClientName"]==clientName) & (client_contract["OAIC"]==oaic)]
    else:
        clientName = f"Nutriset_{client}"
        corresponding_row = client_contract[(client_contract["ClientName"]==clientName)]

    CustomerCode, Contractcode, QuotationCode = list(corresponding_row["CustomerCode"])[0], list(corresponding_row["ContractCode"])[0], list(corresponding_row["QuotationCode"])[0]

    if len(corresponding_row) == 1:
        sample_clean_dict["Name"] = clientName
        sample_clean_dict["CustomerCode"] = CustomerCode
        sample_clean_dict["ContractCode"] = Contractcode
        sample_clean_dict["QuotationCode"] = QuotationCode
    else :
        sample_clean_dict["CustomerCode"] = ""
        sample_clean_dict["ContractCode"] = ""
        sample_clean_dict["QuotationCode"] = ""
    
    return sample_clean_dict

def convertDictToLIMS(stacked_samples_dict, lims_converter, analysis_lims):

    def _get_packages_tests(analysis, analysis_lims):
        """From the list of the analysis and

        Args:
            analysis (_type_): _description_
            analysis_lims (_type_): _description_

        Returns:
            _type_: _description_
        """

        related_test, all_codes = [], []
        for analyse in analysis:
            all_codes += analysis_lims[analysis_lims["Denomination"]==analyse]["Code"].values.tolist()

        # Add related tests
        related_code_lims = analysis_lims.dropna(subset="Related")

        # First all imposed tests
        everytime_list = related_code_lims[related_code_lims["Related"]=="EVERYTIME"]["Code"].values.tolist()
        if everytime_list:
            related_test+=everytime_list

        # All tests that depend on one test
        one_related = related_code_lims[related_code_lims["Related"].str.startswith("ONE")]
        for code in all_codes:
            test = one_related[one_related["Related"].str.contains(code)]["Code"].values.tolist()
            if test:
                related_test += test

        # All tests that depend on several tests
        all_related = related_code_lims[related_code_lims["Related"].str.startswith("ALL")]
        if len(all_related)>0:
            all_related.loc[:,["Related"]] =  deepcopy(all_related["Related"]).apply(lambda x: x.split(": ")[1].split(" "))

        for ind in all_related.index:
            if set(all_related["Related"][ind]).issubset(set(all_codes)):
                related_test.append(all_related["Code"][ind])

        all_codes += related_test
        all_codes = list(set(all_codes))

        customer_package, package_codes, test_codes = [], [], []
        for code in all_codes:
            if len(code)<=4:
                customer_package.append(code)
            elif code[0] == "P":
                package_codes.append(code)
            else:
                test_codes.append(code)

        return package_codes, test_codes, customer_package   

    # The return 
    xmls_format_dict = []

    for sample_dict in stacked_samples_dict:

        sample_XML_dict = {}

        # Give all new items to the sample dict
        package_code, test_code, customer_package = _get_packages_tests(sample_dict["analyse"], analysis_lims)

        if package_code:
            sample_dict["PackageCode"] = package_code
        if test_code:
            sample_dict["TestCode"] = test_code
        if customer_package:
            sample_dict["CustomerPackage"] = customer_package

        # Add the quotation code to the customer package
        QuotationCode = sample_dict["QuotationCode"]
        sample_dict["CustomerPackage"] = [QuotationCode+"."+pack for pack in customer_package]

        input_keys = list(sample_dict.keys())
        
        for name, convert_dict in lims_converter.items():
            value = None
            path, input = convert_dict["path"], convert_dict["input"]
            keys_path = path.split(".")
            if input in input_keys:
                value = sample_dict[input]

            elif type(input) == type([]):
                value = convert_dict["join"].join([sample_dict[inp] for inp in input])

            elif input == "HARDCODE":
                value = convert_dict["value"]
            
            if value:
                _update_dict(sample_XML_dict, value, keys_path)
        
        # Warning if not client or contract
        if not _get_dict_value(sample_XML_dict, "Order.CustomerCode") or not _get_dict_value(sample_XML_dict, "Order.ContractCode"):
            customerRef =  _get_dict_value(sample_XML_dict, "Order.Samples.Sample.CustomerReference")
            if customerRef:
                print(f"PAS DE CLIENT TROUVE POUR : {customerRef}, A CODER MANUELLEMENT")
            else :
                print(f"PAS DE CLIENT TROUVE")

        else:
            xmls_format_dict.append(sample_XML_dict)
        
    return xmls_format_dict

def arrangeForClientSpecificites(stacked_samples_dict, analysis_lims, model):

    def _split_analysis(stacked_samples_dict, analysis_lims):
        
        res_saple_dict = []

        for sample_dict in stacked_samples_dict:
            # Iterate over the copy of the dict which is not going to change
            sample_dict_copy = deepcopy(sample_dict)
            
            # Make n sample for n type of analysis
            for type in analysis_lims["Type"].unique():
                analysis_by_type = analysis_lims[analysis_lims["Type"]==type]["Denomination"].values.tolist()
                sample_dict_copy["analyse"] = list(set(analysis_by_type) & set(sample_dict["analyse"]))
                # If no analysis of the type
                if sample_dict_copy["analyse"]:
                    res_saple_dict.append(deepcopy(sample_dict_copy))

        return res_saple_dict
    
    if "CU" in model:
        stacked_samples_dict =  _split_analysis(stacked_samples_dict, analysis_lims)

    return stacked_samples_dict

def mergeOrderSamples(stacked_samples_dict, merge_condition="Order.PurchaseOrderReference"):
    
    def _merge_bool(merged_dict, sample_dict, merge_condition):
        # 
        if type(merge_condition) == type([]):
            return all([_get_dict_value(merged_dict, condi) == _get_dict_value(sample_dict, condi) for condi in merge_condition])
        else :
            return _get_dict_value(merged_dict, merge_condition) == _get_dict_value(sample_dict, merge_condition)

    stacked_merged_dict = []
    added_number = []
    for sample_dict in stacked_samples_dict:
        merged = False
        for i_dict, merged_dict in enumerate(stacked_merged_dict):
            # Merge samples by common
            if _merge_bool(merged_dict, sample_dict, merge_condition):
                new_number = len(_get_dict_value(merged_dict, "Order.Samples"))
                # Store to clean the xml after the conversion
                added_number.append(new_number)
                # Generete de new dict path
                new_path = f"Order.Samples.Sample_"+str(new_number)
                _update_dict(stacked_merged_dict[i_dict], _get_dict_value(sample_dict, "Order.Samples.Sample"), new_path)
                merged = True

        if not merged:
            stacked_merged_dict.append(sample_dict)

    return stacked_merged_dict, added_number

def finalSaveDict(verified_dict, xmls_save_path, analysis_lims, model, lims_helper, client_contract, xml_name="verified_XML"):

    def _rename_sample(xml, added_number):
        xml = xml.decode("UTF-8")
        for number in added_number:
            # XML accepts to have several time the same keys, dict don't, then sample are normalised
            if f'<Sample_{number} type="dict">' in xml:
                xml = xml.replace(f'<Sample_{number} type="dict">', '<Sample type="dict">')
                xml = xml.replace(f'</Sample_{number}>', '</Sample>')
        xml = xml.encode("UTF-8")
        return xml
    
    # For all sample to extract from pdfs, keep only relevant fields
    stacked_samples_dict = []
    for pdf_name, sample_dict in verified_dict.items():
        for sample, res_dict in sample_dict.items():
            xml_name = datetime.now().strftime("%Y%m%d%H")
            sample_XML_dict = keepNeededFields(res_dict["EXTRACTION"], client_contract, model, lims_helper["PRODUCTCODE_DICT"])
            stacked_samples_dict.append(sample_XML_dict)

    # For all extracted dict, arrage them according to the client/labs needs
    stacked_samples_dict = arrangeForClientSpecificites(stacked_samples_dict, analysis_lims, model)

    # Convert samples dict to the XML format
    lims_converter = lims_helper["LIMS_CONVERTER"]["CU"] if "CU" in model else lims_helper["LIMS_CONVERTER"][model]
    xmls_format_dict = convertDictToLIMS(stacked_samples_dict, lims_converter, analysis_lims)

    xmls_merged_dict, added_number = mergeOrderSamples(xmls_format_dict, merge_condition=lims_helper["SAMPLE_MERGER"])
    
    #######

    # Then convert each dict as XML
    xml_names = []
    for sample_dict in xmls_merged_dict:
        xml_name = "_".join([_get_dict_value(sample_dict, "Order.RecipientLabCode"), pdf_name, datetime.today().strftime('%Y%m%d')])
        # Set the name of the XML : BU_REF_YYMMDD_N
        num = 0
        sample_XML_num = xml_name+f"_{num}"
        while sample_XML_num in xml_names:
            num+=1
            sample_XML_num = xml_name+f"_{num}"
        xml_names.append(sample_XML_num)

        # Create the XML
        xml = dicttoxml.dicttoxml(sample_dict)
        xml = _rename_sample(xml, added_number)
        xml_save_path = os.path.join(xmls_save_path, f"{sample_XML_num}.xml")
        with open(xml_save_path, 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())

def saveToCopyFolder(save_folder, pdf_path, rename="", mode="same"):
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        base, extension = os.path.splitext(os.path.split(pdf_path)[1])

        if rename:
            base=rename
        
        if mode == "same":
            new_name = base+extension

        copyfile(pdf_path, f"{save_folder}/{new_name}")

if __name__ == "__main__":
    import pandas as pd

    verified_dict = {'CU OAIC': 
                     {'sample_0': {'IMAGE': 'image_0', 
                                   'EXTRACTION': {'reference': 
                                                  {'sequence': 'RAF', 'confidence': 0.0, 'area': [1240, 0, 2480, 1052]}, 
                                                  'navire': {'sequence': 'M/V - LUTA', 'confidence': 0.959, 'area': [254, 600, 1872, 721]}, 
                                                  'marchandise': {'sequence': 'BLE TENDRE', 'confidence': 0.984, 'area': [286, 655, 1739, 776]}, 
                                                  'quantite': {'sequence': '31 500 MT', 'confidence': 1.0, 'area': [398, 720, 1509, 823]}, 
                                                  'vendeur': {'sequence': 'AVERE', 'confidence': 0.999, 'area': [406, 761, 1496, 892]}, 
                                                  'date': {'sequence': '05/02/2024', 'confidence': 0.997, 'area': [374, 814, 2395, 955]}, 
                                                  'port': {'sequence': 'ROUEN', 'confidence': 0.956, 'area': [1066, 602, 2480, 726]}, 
                                                  'destination': {'sequence': 'ALGERIE', 'confidence': 0.999, 'area': [1272, 646, 2480, 798]}, 
                                                  'acheteur': {'sequence': 'OAIC', 'confidence': 1.0, 'area': [1362, 779, 2480, 883]}, 
                                                  'scelles': {'sequence': 'CUF 15', 'confidence': 0.924, 'area': [1400, 823, 2480, 954]}, 
                                                  'analyse': {'sequence': ['Pack Protéines Dumas sec x5,7 + Humidité', "Zéaralénone",'Alvéogramme', 'Hagberg M0270', 'Poids spécifique QF015', 'GRAINS MOUCHETES / BOUTES / COLORES', 'Impuretés sur blé (ISO 7970)', 'ZELENY'], 'confidence': 0.981, 'area': [0, 701, 2480, 3508]}, 
                                                  'code_produit': {'sequence': 'Ble tendre', 'confidence': 0.984, 'area': [286, 655, 1739, 776]}, 'client': 'Rouen'}}}}
                                                  
                                                  
    OCR_HELPER = json.load(open("CONFIG\OCR_config.json"))

    client_contract =  pd.read_excel(r"CONFIG\\eLIMS_contract_analysis.xlsx")
    xml_save_path = r"C:\Users\CF6P\Desktop\ECAR\Data\debug"
    model = "CU OAIC"
    analysis_lims = pd.read_excel(OCR_HELPER["PATHES"]["contract_analysis_path"], sheet_name="analyse"+" "+model)
    lims_converter =  json.load(open("CONFIG\LIMS_CONFIG.json"))

    finalSaveDict(verified_dict, xml_save_path, analysis_lims, model, lims_converter, client_contract, xml_name="verified_XML")

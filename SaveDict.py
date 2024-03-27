import os
import dicttoxml
import json
import re
from copy import deepcopy
from datetime import datetime

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
        value = value[key]
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

        # Product Code case
        elif key[0] == "code_produit":
            if items:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[0]]["sequence"] = key[1]

        # Other cases
        elif key[0] == "zone":
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[1]]["sequence"] = items


    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse"]["sequence"] = analyses
    # res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse_specification"] = {"sequence" : comments}

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
    
    return res_dict

def keepNeededFields(verified_dict, client_contract, model):
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
    if "code_produit" in sample_clean_dict.keys():
        sample_clean_dict["code_produit"] = verified_dict["code_produit"]["sequence"].split(": ")[1] if verified_dict["code_produit"]["sequence"] else ""

    # Find the Contract and the quotation according to the data
    client = verified_dict["client"]
    clientName = "CU" if "CU" in model else model
    oaic = "Oui" if model == "CU OAIC" else "Non"

    corresponding_row = client_contract[(client_contract["ClientName"]==clientName) & (client_contract["Client"]==client) & (client_contract["OAIC"]==oaic)]
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
        related_test.append(related_code_lims[related_code_lims["Related"]=="EVERYTIME"]["Code"].values.tolist()[0])

        # All tests that depend on one test
        one_related = related_code_lims[related_code_lims["Related"].str.contains("ONE")]
        for code in all_codes:
            test = one_related[one_related["Related"].str.contains(code)]["Code"].values.tolist()
            if test:
                if not test[0] in related_test:
                    related_test.append(test[0])

        # All tests that depend on several tests
        all_related = related_code_lims[related_code_lims["Related"].str.contains("ALL")]
        if len(all_related)>0:
            all_related.loc[:,["Related"]] =  deepcopy(all_related["Related"]).apply(lambda x: x.split(": ")[1].split(" "))

        for ind in all_related.index:
            if set(all_related["Related"][ind]).issubset(set(all_codes)):
                related_test.append(all_related["Code"][ind])

        related_test = list(set(related_test))

        all_codes += related_test
        customer_package, package_codes, test_codes = [], [], []
        for code in all_codes:
            if len(code)<5:
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

def finalSaveDict(verified_dict, xmls_save_path, analysis_lims, model, lims_converter, client_contract, xml_name="verified_XML"):

    def _rename_sample(xml, added_number):
        xml = xml.decode("UTF-8")
        for number in added_number:
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
            sample_XML_dict = keepNeededFields(res_dict["EXTRACTION"], client_contract, model)
            stacked_samples_dict.append(sample_XML_dict)

    # For all extracted dict, arrage them according to the client/labs needs
    # stacked_samples_dict = arrangeForClientSpecificites(stacked_samples_dict, analysis_lims, model)

    # Convert samples dict to the XML format
    xmls_format_dict = convertDictToLIMS(stacked_samples_dict, lims_converter, analysis_lims)

    xmls_merged_dict, added_number = mergeOrderSamples(xmls_format_dict)
    
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

if __name__ == "__main__":
    import pandas as pd

    verified_dict = {'Control2': {'sample_0': {'IMAGE': 'image_0', 'EXTRACTION': {'reference': {'sequence': 'REF1', 'confidence': 0.0, 'area': [1240, 0, 2480, 1052]}, 'navire': {'sequence': 'M/V - SINCERE', 'confidence': 0.989, 'area': [95, 403, 1720, 534]}, 'marchandise': {'sequence': 'BLE TENDRE', 'confidence': 0.998, 'area': [136, 472, 1568, 576]}, 'quantite': {'sequence': '27400,000 MT', 'confidence': 0.999, 'area': [228, 516, 1411, 647]}, 'vendeur': {'sequence': 'VITERRA BV', 'confidence': 1.0, 'area': [1358, 534, 2448, 638]}, 'date': {'sequence': '09/02/2024', 'confidence': 1.0, 'area': [255, 575, 1342, 706]}, 'port': {'sequence': 'CHORNOMORSK / UKRAINE', 'confidence': 0.988, 'area': [1047, 412, 2480, 536]}, 'destination': {'sequence': 'TUNISE', 'confidence': 1.0, 'area': [1263, 479, 2480, 583]}, 'acheteur': {'sequence': 'OFFICE DES CEREALES', 'confidence': 0.999, 'area': [1343, 593, 2480, 697]}, 'scelles': {'sequence': '3942668', 'confidence': 0.999, 'area': [1324, 757, 2480, 888]}, 'analyse': {'sequence': ['Poids spécifique ', 'Pesticides (Pack) CODEX Autres', 'Deltaméthrin',  'Cyanures'], 'confidence': 0.941, 'area': [0, 701, 2480, 3508]}, 'code_produit': {'sequence': '06161: Blé tendre', 'confidence': 0.998, 'area': [136, 472, 1568, 576]}, 'client': 'Rouen'}}, 'sample_1': {'IMAGE': 'image_1', 'EXTRACTION': {'reference': {'sequence': 'REF2', 'confidence': 0.0, 'area': [1240, 0, 2480, 1052]}, 'navire': {'sequence': 'M/V - ARKLOW WIND', 'confidence': 0.985, 'area': 
                [16, 306, 1766, 447]}, 'marchandise': {'sequence': 'BLE TENDRE', 'confidence': 0.988, 'area': [61, 377, 1607, 498]}, 'quantite': {'sequence': '15 080 MT 000', 'confidence': 1.0, 'area': [164, 428, 1406, 562]}, 'vendeur': {'sequence': 'AGRIAL', 'confidence': 1.0, 'area': [1341, 450, 2480, 554]}, 'date': {'sequence': '06/02/2024', 'confidence': 1.0, 'area': [184, 487, 1367, 618]}, 'port': {'sequence': 'CAEN', 'confidence': 1.0, 'area': 
                [1014, 317, 2480, 458]}, 'destination': {'sequence': 'UK', 'confidence': 1.0, 'area': [1246, 392, 2480, 496]}, 'acheteur': {'sequence': '', 'confidence': 0.0, 'area': [0, 70, 2480, 1052]}, 'scelles': {'sequence': 'AUTRES', 'confidence': 0.998, 'area': [1370, 498, 2480, 629]}, 'analyse': {'sequence': ['Humidité uniquement (AA07A)', 'Impuretés ISO15587 (QF04M)', 'Insectes vivants QF045', 'Insectes mort QF044', 'Zéaralénone immuno', 'Deltaméthrin'], 'confidence': 0.519, 'area': [0, 701, 2480, 3508]}, 'code_produit': {'sequence': '06161: Blé tendre', 'confidence': 0.988, 'area': [61, 377, 1607, 498]}, 'client': 'Rouen'}}}}

    OCR_HELPER = json.load(open("CONFIG\OCR_config.json"))


    client_contract =  pd.read_excel(r"CONFIG\\client_contract.xlsx")
    xml_save_path = r"C:\Users\CF6P\Desktop\ECAR\Data\debug"
    model = "CU hors OAIC"
    analysis_lims = pd.read_excel(OCR_HELPER["analysis_path"], sheet_name=model)
    lims_converter =  json.load(open("CONFIG\LIMS_CONFIG.json"))["LIMS_CONVERTER"]

    finalSaveDict(verified_dict, xml_save_path, analysis_lims, model, lims_converter, client_contract, xml_name="verified_XML")

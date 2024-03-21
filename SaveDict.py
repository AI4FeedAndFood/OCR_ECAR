import os
import dicttoxml
import json
from copy import deepcopy
from datetime import datetime

def runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows):

    analyses, comments = [], []

    for key, items in verif_values.items():
        if key[0] == "ana":
            index = key[1]
            if not index in deleted_rows:
                analyses.append(items)
                # comments.append(verif_values[("spec", index)])

            elif "spec" in key:
                continue

        elif key[0] == "zone":
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[1]]["sequence"] = items

        elif key[0] == "client":
            if items:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[0]] = key[1]

    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse"]["sequence"] = analyses
    # res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse_specification"] = {"sequence" : comments}

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
    
    return res_dict

def convertDictToLIMS(verified_dict, analysis_lims, lims_converter, client_contract, model): 
    
    def _get_packages_tests(analysis, analysis_lims):

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
        related_col = deepcopy(all_related["Related"])
        related_col = related_col.apply(lambda x: x.split(": ")[1].split(" "))
        all_related.loc[:,["Related"]] = related_col

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
    
    def _update_dict(stack_dict, value, keys_path):
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

    added_field = ["analyse_specification", "client", "code_produit"]

    # dict for clean fields
    scan_clean_dict = {}
    for key, value  in verified_dict.items():
        if not key in added_field:
            scan_clean_dict[key] = value["sequence"]

    # Add the code_produit
    scan_clean_dict["code_produit"] = verified_dict["code_produit"]["sequence"].split(":")[0] if verified_dict["code_produit"]["sequence"] else ""

    # Find the Contract and the quotation according to the data
    client = verified_dict["client"]
    clientName = "CU" if "CU" in model else model
    oaic = "Oui" if model == "CU OAIC" else "Non"

    corresponding_row = client_contract[(client_contract["ClientName"]==clientName) & (client_contract["Client"]==client) & (client_contract["OAIC"]==oaic)]
    CustomerCode, Contractcode, QuotationCode = list(corresponding_row["CustomerCode"])[0], list(corresponding_row["ContractCode"])[0], list(corresponding_row["QuotationCode"])[0]

    if len(corresponding_row) == 1:
        scan_clean_dict["CustomerCode"] = CustomerCode
        scan_clean_dict["ContractCode"] = Contractcode
        scan_clean_dict["QuotationCode"] = QuotationCode
    else :
        scan_clean_dict["CustomerCode"] = ""
        scan_clean_dict["ContractCode"] = ""
        scan_clean_dict["QuotationCode"] = ""

    package_code, test_code, customer_package = _get_packages_tests(verified_dict["analyse"]["sequence"], analysis_lims)
    customer_package = [QuotationCode+"."+pack for pack in customer_package]

    if package_code:
        scan_clean_dict["PackageCode"] = package_code
    if test_code:
        scan_clean_dict["TestCode"] = test_code
    if customer_package:
        scan_clean_dict["CustomerPackage"] = customer_package

    # Initialized all layers of the res XML
    stack_dict = {}

    input_keys = list(scan_clean_dict.keys())

    for name, convert_dict in lims_converter.items():
        value = None
        path, input = convert_dict["path"], convert_dict["input"]
        keys_path = path.split(".")
        if input in input_keys:
            value = scan_clean_dict[input]

        elif type(input) == type([]):
            value = convert_dict["join"].join([scan_clean_dict[inp] for inp in input])

        elif input == "HARDCODE":
            value = convert_dict["value"]
        
        if value:
            _update_dict(stack_dict, value, keys_path)
    
    return stack_dict

def finalSaveDict(verified_dict, xml_save_path, analysis_lims, model, lims_converter, client_contract, out_path="", xml_name="verified_XML"):

    def _copy_dict(verified_dict):
        xml_dict = {}
        num = 1
        for pdf_name, sample_dict in verified_dict.items():
            for sample, res_dict in sample_dict.items():
                xml_name = datetime.now().strftime("%Y%m%d%H")
                clean_dict = convertDictToLIMS(res_dict["EXTRACTION"], analysis_lims, lims_converter, client_contract, model)
                sample_num_id = xml_name+f"_{num}"
                while sample_num_id in list((xml_dict).keys()):
                    num+=1
                    sample_num_id = xml_name+f"_{num}"
                xml_dict[sample_num_id] = deepcopy(clean_dict)
                # res_dict[scan_name] = clean_dict

        return xml_dict
    
    if out_path:
        new_xml = os.path.join(out_path, xml_name)
        if not os.path.exists(new_xml):
            os.makedirs(new_xml)
    
    xml_dict = _copy_dict(verified_dict)
    xml_dict = dict(sorted(xml_dict.items()))

    for scan_name, scan_dict in xml_dict.items():
        xml = dicttoxml.dicttoxml(scan_dict)
        with open(os.path.join(xml_save_path, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())
        if out_path:
            with open(os.path.join(new_xml, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
                result_file.write(xml.decode())

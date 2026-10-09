

def get_data_info(data_name):
    if data_name == "mayo2016_3mm" or data_name == "mayo2016_1mm" or data_name == "mayo2016_3mm_1q2" or data_name == "mayo2016_3mm_1q10" or data_name == "mayo2016_3mm_1q20":
        if data_name == "mayo2016_3mm": data_path = "/data/xcshen/data/mayo2016_3mm_processed_data"
        if data_name == "mayo2016_1mm": data_path = "/data/xcshen/data/mayo2016_1mm_processed_data"
        if data_name == "mayo2016_3mm_1q2": data_path = "/data/xcshen/data/mayo2016_3mm_1q2_processed_data"
        if data_name == "mayo2016_3mm_1q10": data_path = "/data/xcshen/data/mayo2016_3mm_1q10_processed_data"
        if data_name == "mayo2016_3mm_1q20": data_path = "/data/xcshen/data/mayo2016_3mm_1q20_processed_data"
        train_ids = ["L067", "L096", "L109", "L143", "L192", "L286", "L291"]
        val_ids = ["L310"]
        test_ids = ["L333", "L506"]
        
    if data_name == "mayo2020_5mm":
        data_path = "/data/xcshen/data/mayo2020_5mm_processed_data"
        train_ids = ['L004', 'L006', 'L014', 'L019', 'L033', 'L049', 'L056', 'L057', 'L058', 'L064', 'L071', 'L072', 'L075', 'L077', 'L081', 'L107', 'L110', 'L114', 'L116', 'L123', 'L125', 'L131', 'L134', 'L145', 'L148', 'L150', 'L160', 'L170', 'L175', 'L178', 'L179', 'L186', 'L187', 'L193', 'L203', 'L209', 'L210', 'L212', 'L219', 'L220', 'L221', 'L229', 'L232', 'L237', 'L241', 'L248', 'L266']
        val_ids = ['L273']
        test_ids = ['L277', 'L299']
    
    if data_name == "mayo2020_1d5mm":
        data_path = "/data/xcshen/data/mayo2020_1d5mm_processed_data"
        train_ids = ['C002', 'C004', 'C012', 'C016', 'C021', 'C027', 'C030', 'C050', 'C052', 'C067', 'C077', 'C081', 'C095', 'C099', 'C107', 'C111', 'C120', 'C121', 'C124', 'C128', 'C130', 'C135', 'C158', 'C160', 'C162']
        val_ids = ['C295']
        test_ids = ['C296']
    
    if data_name == "siemens_small":
        data_path = "/data/xcshen/data/siemens_processed_data_small"
        train_ids = [f"siemens{ID}" for ID in range(1, 23)]
        val_ids = ["siemens23"]
        test_ids = ["siemens24", "siemens25"]
    
    if data_name == "siemens_large":
        data_path = "/data/xcshen/data/siemens_processed_data_large"
        train_ids = [f"siemens{idx}" for idx in range(1, 46)]
        val_ids = ["siemens46"]
        test_ids = ["siemens47"]
    
    if data_name == "mvct_kvct":
        data_path = "/data/xcshen/data/mvct_kvct_processed_data"
        train_ids = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P14', 'P15', 'P16', 'P17', 'P19', 'P20', 'P21', 'P22', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P35']
        val_ids = ["P36"]
        test_ids = ["P37", "P38"]
    
    if data_name == "motif":
        data_path = "/data/xcshen/data/motif_processed_data"
        train_ids = []
        val_ids = []
        test_ids = ["motif"]
    
    return data_path, train_ids, val_ids, test_ids


if __name__ == "__main__":
    import re, os
    from natsort import natsorted
    
    data_name = "mayo2016_3mm"
    data_path, train_ids, val_ids, test_ids = get_data_info(data_name)
    
    train_re = re.compile(rf"({'|'.join(train_ids)})_[a-zA-Z0-9]+_input.npy")
    val_re   = re.compile(rf"({'|'.join(val_ids)})_[a-zA-Z0-9]+_input.npy")
    test_re  = re.compile(rf"({'|'.join(test_ids)})_[a-zA-Z0-9]+_input.npy")
    
    all_file = os.listdir(data_path)
    
    train_list = natsorted([x for x in all_file if train_re.match(x)])
    val_list = natsorted([x for x in all_file if val_re.match(x)])
    test_list = natsorted([x for x in all_file if test_re.match(x)])
    
    print(f"dataset: {data_name}")
    print(f"Number of volumes: {len(train_ids) + len(val_ids) + len(test_ids)}, Number of slices: {len(train_list) + len(val_list) + len(test_list)}")
    print(f"Number of training volumes: {len(train_ids)}, Number of slices: {len(train_list)}")
    print(f"Number of validation volumes: {len(val_ids)}, Number of slices: {len(val_list)}")
    print(f"Number of test volumes: {len(test_ids)}, Number of slices: {len(test_list)}")
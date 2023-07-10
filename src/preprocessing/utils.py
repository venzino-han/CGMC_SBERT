from tqdm import tqdm

def convert_newid(origin_id:int, id_dict:dict, max_id:int):

    if origin_id in id_dict :
        new_id = id_dict.get(origin_id)
    else:
        id_dict[origin_id] = max_id
        new_id = max_id
        max_id += 1

    return new_id, id_dict, max_id


def reset_id(df, user_col, item_col, cols):

    user_id_dict, item_id_dict, user_ids, item_ids, user_id_max, item_id_max  = {}, {}, [], [], 0, 0        
    
    for i in tqdm(range(len(df))):
        origin_user_id = df[user_col].iloc[i]
        origin_item_id = df[item_col].iloc[i]
        new_user_id, user_id_dict, user_id_max = convert_newid(origin_user_id, user_id_dict, user_id_max)
        new_item_id, item_id_dict, item_id_max = convert_newid(origin_item_id, item_id_dict, item_id_max)

        user_ids.append(new_user_id)
        item_ids.append(new_item_id)

    df[user_col] = user_ids
    df[item_col] = item_ids

    df = df[cols]
    
    return df
import os
import h5py
import ast

def load_mapping_dict(file_path):
    with h5py.File(file_path, 'r') as file:
        mapping_bytes = file['Mapping_dict'][()]
        mapping_str = mapping_bytes.decode('utf-8')
        mapping_dict = ast.literal_eval(mapping_str)
    return mapping_dict

def save_mapping_dict(file_path, new_dict):
    with h5py.File(file_path, 'a') as file:
        # 將新字典轉換為字節串後保存
        new_mapping_bytes = str(new_dict).encode('utf-8')
        if 'Mapping_dict' in file:
            del file['Mapping_dict']  # 刪除原來的 Mapping_dict
        file.create_dataset('Mapping_dict', data=new_mapping_bytes)

def reset_and_add_gestures(file_path, gestures):
    new_dict = {'0': 'Background'}
    for i, (gesture_id, gesture_name) in enumerate(gestures.items(), start=1):
        new_dict[str(i)] = gesture_name
    save_mapping_dict(file_path, new_dict)
    print("Gestures reset and added successfully.")
    print("New Mapping Dict:", new_dict)

# 獲取當前腳本的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'K60168-Release-01018-002-v0.0.2-20240613.h5')

# 檢查檔案是否存在
if os.path.exists(file_path):
    print("File found!")
else:
    print("File not found at:", file_path)
    exit(1)

# 定義要新增的手勢（字典形式，鍵為手勢編號，值為手勢名稱）
new_gestures = {
    '1': 'Normal',
    '2': 'Fast',
    '3': 'SuperFast',
    '4': 'WrongTask'
}

# 重置並新增手勢
reset_and_add_gestures(file_path, new_gestures)

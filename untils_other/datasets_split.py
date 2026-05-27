import json
import random
import os

def create_mini_splits(data_root, output_root, ratio=0.5, seed=42):
    """
    data_root: 原始 json 存放路径
    output_root: mini json 输出路径
    ratio: 抽取比例 (0.2)
    """
    # 设定随机种子，保证 Baseline 和你的模型用的是同一份数据
    random.seed(seed)
    
    # 定义你要处理的文件
    train_files = ['scanrefer_mask3d_train.json', 
                  'scanqa_train.json',
                  'obj_align_mask3d_train.json',
                  'nr3d_caption_mask3d_train.json',
                  'object_descriptions_with_obj_tags.json']
    
    val_test_files = ['scanrefer_mask3d_val.json',
                     'scanqa_val.json',
                     'obj_align_mask3d_val.json',
                     'scanrefer_mask3d_test.json', 
                     'scanqa_test.json']
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # 1. 处理训练数据集 - 确保使用相同的场景ID
    print("="*50)
    print("处理训练数据集 - 确保使用相同的场景ID")
    print("="*50)
    
    # 收集所有训练数据集的场景ID
    all_train_scene_sets = []
    train_scene_counts = {}
    
    for file_name in train_files:
        file_path = os.path.join(data_root, file_name)
        if not os.path.exists(file_path):
            print(f"跳过: 找不到文件 {file_path}")
            continue
            
        print(f"正在分析: {file_name} ...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 获取该数据集的所有唯一scene_id
        scenes = set(item['scene_id'] for item in data)
        all_train_scene_sets.append(scenes)
        train_scene_counts[file_name] = len(scenes)
    
    # 找出所有训练数据集共有的场景ID
    if all_train_scene_sets:
        common_train_scenes = set.intersection(*all_train_scene_sets)
        print(f"\n所有训练数据集共有的场景ID数量: {len(common_train_scenes)}")
        
        # 从共有场景中随机抽取指定比例
        common_train_scenes_list = sorted(list(common_train_scenes))  # 排序保证结果可复现
        num_sample = int(len(common_train_scenes_list) * ratio)
        num_sample = max(1, num_sample)  # 保证至少抽1个场景
        
        selected_train_scenes = set(random.sample(common_train_scenes_list, num_sample))
        print(f"从共有场景中随机抽取了 {len(selected_train_scenes)} 个场景ID\n")
        
        # 使用选中的场景ID过滤所有训练数据集
        for file_name in train_files:
            file_path = os.path.join(data_root, file_name)
            if not os.path.exists(file_path):
                continue
                
            print(f"正在处理: {file_name} ...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 过滤数据：保留属于选中场景的所有样本
            mini_data = [item for item in data if item['scene_id'] in selected_train_scenes]
            
            # 保存
            output_name = file_name.replace('.json', f'_mini_{int(ratio*100)}pct.json')
            output_path = os.path.join(output_root, output_name)
            
            with open(output_path, 'w') as f:
                json.dump(mini_data, f)
                
            print(f"  - 原始场景数: {train_scene_counts[file_name]} -> 抽取: {len(set(item['scene_id'] for item in mini_data))}")
            print(f"  - 原始样本数: {len(data)} -> 抽取: {len(mini_data)}")
            print(f"  - 已保存至: {output_path}\n")
    else:
        print("没有找到任何有效的训练数据集文件")
    
    # 2. 处理验证和测试数据集 - 独立随机抽取
    print("="*50)
    print("处理验证和测试数据集 - 独立随机抽取")
    print("="*50)
    
    for file_name in val_test_files:
        file_path = os.path.join(data_root, file_name)
        if not os.path.exists(file_path):
            print(f"跳过: 找不到文件 {file_path}")
            continue
            
        print(f"正在处理: {file_name} ...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 获取该数据集的所有唯一scene_id
        all_scenes = list(set([item['scene_id'] for item in data]))
        all_scenes.sort() # 排序保证结果可复现
        
        # 按SceneID随机抽取指定比例
        num_sample = int(len(all_scenes) * ratio)
        num_sample = max(1, num_sample)  # 保证至少抽1个场景
        
        selected_scenes = set(random.sample(all_scenes, num_sample))
        
        # 过滤数据：保留属于选中场景的所有样本
        mini_data = [item for item in data if item['scene_id'] in selected_scenes]
        
        # 保存
        output_name = file_name.replace('.json', f'_mini_{int(ratio*100)}pct.json')
        output_path = os.path.join(output_root, output_name)
        
        with open(output_path, 'w') as f:
            json.dump(mini_data, f)
            
        print(f"  - 原始场景数: {len(all_scenes)} -> 抽取: {len(selected_scenes)}")
        print(f"  - 原始样本数: {len(data)} -> 抽取: {len(mini_data)}")
        print(f"  - 已保存至: {output_path}\n")

def main():
    create_mini_splits('/home/lcx/chat-scene/Chat-Scene/annotations/', '/home/lcx/chat-scene/Chat-Scene/annotations/')

if __name__ == "__main__":
    main()

"""
==================================================
处理训练数据集 - 确保使用相同的场景ID
==================================================
正在分析: scanrefer_mask3d_train.json ...
正在分析: scanqa_train.json ...
正在分析: obj_align_mask3d_train.json ...
正在分析: nr3d_caption_mask3d_train.json ...
正在分析: object_descriptions_with_obj_tags.json ...

所有训练数据集共有的场景ID数量: 507
从共有场景中随机抽取了 253 个场景ID

正在处理: scanrefer_mask3d_train.json ...
  - 原始场景数: 561 -> 抽取: 253
  - 原始样本数: 35061 -> 抽取: 16627
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanrefer_mask3d_train_mini_50pct.json

正在处理: scanqa_train.json ...
  - 原始场景数: 562 -> 抽取: 253
  - 原始样本数: 26138 -> 抽取: 12329
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanqa_train_mini_50pct.json

正在处理: obj_align_mask3d_train.json ...
  - 原始场景数: 1199 -> 抽取: 253
  - 原始样本数: 26010 -> 抽取: 5387
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/obj_align_mask3d_train_mini_50pct.json

正在处理: nr3d_caption_mask3d_train.json ...
  - 原始场景数: 509 -> 抽取: 253
  - 原始样本数: 30265 -> 抽取: 15419
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/nr3d_caption_mask3d_train_mini_50pct.json

正在处理: object_descriptions_with_obj_tags.json ...
  - 原始场景数: 1510 -> 抽取: 253
  - 原始样本数: 137771 -> 抽取: 22843
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/object_descriptions_with_obj_tags_mini_50pct.json

==================================================
处理验证和测试数据集 - 独立随机抽取
==================================================
正在处理: scanrefer_mask3d_val.json ...
  - 原始场景数: 141 -> 抽取: 70
  - 原始样本数: 9508 -> 抽取: 4615
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanrefer_mask3d_val_mini_50pct.json

正在处理: scanqa_val.json ...
  - 原始场景数: 71 -> 抽取: 35
  - 原始样本数: 4675 -> 抽取: 2288
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanqa_val_mini_50pct.json

正在处理: obj_align_mask3d_val.json ...
  - 原始场景数: 312 -> 抽取: 156
  - 原始样本数: 7839 -> 抽取: 3733
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/obj_align_mask3d_val_mini_50pct.json

正在处理: scanrefer_mask3d_test.json ...
  - 原始场景数: 97 -> 抽取: 48
  - 原始样本数: 5410 -> 抽取: 2760
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanrefer_mask3d_test_mini_50pct.json

正在处理: scanqa_test.json ...
  - 原始场景数: 167 -> 抽取: 83
  - 原始样本数: 11125 -> 抽取: 5209
  - 已保存至: /home/lcx/chat-scene/Chat-Scene/annotations/scanqa_test_mini_50pct.json
"""
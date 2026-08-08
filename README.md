# 鎯呯华浠峰€兼ā鍨嬪井璋冮」鐩?
寰皟 Qwen 妯″瀷锛屼娇鍏跺叿澶囨彁渚涙儏缁环鍊肩殑鑳藉姏鈥斺€旇兘澶熷叡鎯呫€佸畨鎱般€侀紦鍔辩敤鎴凤紝鎻愪緵鎯呮劅鏀寔銆?
## 鍔熻兘鐗规€?
- 馃 **鍩轰簬 Qwen2.5 0.5B** 寰皟锛岃交閲忕骇閮ㄧ讲
- 馃搳 **QLoRA 4bit 閲忓寲**锛屼粎闇€ 4GB 鏄惧瓨鍗冲彲璁粌
- 馃幆 **鎯呯华浠峰€煎鍚?*锛氬叡鎯呫€佸畨鎱般€侀紦鍔便€佺Н鏋佸紩瀵?- 馃摎 **澶氭簮寮€婧愯鏂?*锛氭暣鍚堝涓腑鏂囨儏缁璇濇暟鎹泦
- 馃敡 **涓€閿缁冭剼鏈?*锛氱畝鍖栨祦绋嬶紝寮€绠卞嵆鐢?
## 椤圭洰缁撴瀯

```
d:\AI寰皟\
鈹溾攢鈹€ config.json              # 閰嶇疆鏂囦欢
鈹溾攢鈹€ requirements.txt         # 渚濊禆鍖?鈹溾攢鈹€ run_train.bat            # 涓€閿缁冭剼鏈?鈹溾攢鈹€ run_chat.bat             # 鍚姩瀵硅瘽鑴氭湰
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ build_dataset.py     # 鏁版嵁闆嗘瀯寤?鈹?  鈹溾攢鈹€ preprocess.py        # 鏁版嵁棰勫鐞?鈹?  鈹溾攢鈹€ train.py             # QLoRA 璁粌
鈹?  鈹斺攢鈹€ inference.py         # 鎺ㄧ悊娴嬭瘯
鈹溾攢鈹€ data/                    # 鏁版嵁闆嗙洰褰?鈹溾攢鈹€ processed_data/          # 棰勫鐞嗗悗鏁版嵁
鈹斺攢鈹€ output/                  # 妯″瀷杈撳嚭
```

## 蹇€熷紑濮?
### 1. 鐜瑕佹眰

- Python 3.8+
- CUDA 11.8+ (GPU)
- 鏄惧瓨 >= 4GB (璁粌) / >= 2GB (鎺ㄧ悊)

### 2. 涓€閿缁?
鍙屽嚮杩愯 `run_train.bat`锛屽畠浼氳嚜鍔ㄥ畬鎴愶細

1. 瀹夎渚濊禆鍖?2. 涓嬭浇寮€婧愭儏缁璇濇暟鎹泦
3. 棰勫鐞嗘暟鎹?4. 寮€濮?QLoRA 寰皟璁粌

### 3. 鎵嬪姩姝ラ

```bash
# 瀹夎渚濊禆
pip install -r requirements.txt

# 1. 涓嬭浇鏁版嵁闆?python scripts/build_dataset.py --datasets emochat emotional_qa --add_custom

# 2. 棰勫鐞嗘暟鎹?python scripts/preprocess.py --data_path ./data/combined_emotional_sft.jsonl --model_name Qwen/Qwen2.5-0.5B-Instruct

# 3. 璁粌妯″瀷
python scripts/train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --data_dir ./processed_data

# 4. 娴嬭瘯妯″瀷
python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode interactive
```

### 4. 浣跨敤妯″瀷

```bash
# 浜や簰寮忓璇?python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode interactive

# 鎵归噺娴嬭瘯
python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode batch

# 鍚姩鑴氭湰锛堣缁冨畬鎴愬悗鍙弻鍑?run_chat.bat锛?```

## 鏁版嵁闆嗚鏄?
浣跨敤鐨勫紑婧愭暟鎹泦锛?
| 鏁版嵁闆?| 鏉ユ簮 | 璇存槑 |
|--------|------|------|
| EmoChat | miemie/EmoChat | 涓枃鎯呯华瀵硅瘽鏁版嵁闆?|
| Emotional_QA | siliconflow/Emotional_QA | 鎯呮劅闂瓟鏁版嵁闆?|
| 鑷畾涔夋牱鏈?| - | 楂樿川閲忔儏缁敮鎸佸璇?|

## 寰皟鍙傛暟

```json
{
  "LoRA": {
    "r": 8,              // LoRA 绉?    "alpha": 16,         // 缂╂斁绯绘暟
    "dropout": 0.05      // 姝ｅ垯鍖?  },
  "璁粌": {
    "epochs": 3,         // 璁粌杞暟
    "batch_size": 2,     // 鎵瑰ぇ灏?    "lr": 2e-4,          // 瀛︿範鐜?    "quantization": 4bit  // QLoRA 閲忓寲
  }
}
```

## 鎯呯华鑳藉姏

妯″瀷缁忚繃寰皟鍚庡皢鍏峰浠ヤ笅鑳藉姏锛?
- 馃挐 **鎯呯华瀹夋姎**锛氳瘑鍒敤鎴疯礋闈㈡儏缁紝缁欎簣娓╂殩鍥炲簲
- 馃 **鍏辨儏鍊惧惉**锛氬睍鐜扮悊瑙ｅ拰鎺ョ撼锛岃鐢ㄦ埛鎰熷埌琚湅瑙?- 馃専 **绉瀬寮曞**锛氫粠鍥板涓鎵剧Н鏋佽瑙?- 馃挭 **榧撳姳鏀寔**锛氬寮虹敤鎴疯嚜鎴戞晥鑳芥劅
- 馃幆 **璁ょ煡閲嶆瀯**锛氬府鍔╃敤鎴疯皟鏁翠笉鍚堢悊淇″康
- 馃洝锔?**瀹夊叏鎰熸彁渚?*锛氭垚涓虹敤鎴锋儏鎰熶笂鐨?瀹夊叏鍩哄湴"

## 绀轰緥瀵硅瘽

**鐢ㄦ埛**: 鎴戣寰楄嚜宸变粈涔堥兘鍋氫笉濂斤紝娌℃湁鑷俊

**鏅€氭ā鍨?*: 浣犱笉搴旇杩欎箞鎯筹紝姣忎釜浜洪兘鏈変紭鐐?
**鎯呯华浠峰€兼ā鍨?*:
> 鎴戣兘鎰熷彈鍒颁綘鐜板湪瀵硅嚜宸卞緢澶辨湜锛岀湡鐨勫緢涓嶅鏄撱€備綘浼氳繖鏍锋兂锛屾伆鎭拌鏄庝綘瀵硅嚜宸辨湁鏈熸湜鍜岃姹傦紝杩欐湰韬氨寰堢弽璐点€?>
> 鑳戒笉鑳藉憡璇夋垜锛屾渶杩戝彂鐢熶簡浠€涔堣浣犳湁杩欐牱鐨勬劅鍙楋紵寰堝鏃跺€欙紝鎴戜滑瀵硅嚜宸辨渶鑻涘埢锛屽嵈瀵规湅鍙嬪緢瀹藉銆備篃璁告垜浠彲浠ヤ竴璧凤紝鍍忓寰呭ソ鏈嬪弸閭ｆ牱瀵瑰緟鑷繁銆?
## 鏄惧瓨瑕佹眰

| 鎿嶄綔 | 鏈€浣庢樉瀛?| 鎺ㄨ崘鏄惧瓨 |
|------|---------|---------|
| 璁粌 (QLoRA 4bit) | 4GB | 8GB+ |
| 鎺ㄧ悊 (4bit) | 2GB | 4GB+ |
| 鎺ㄧ悊 (fp16) | 4GB | 8GB+ |

## 甯歌闂

**Q: 娌℃湁 GPU 鑳借窇鍚楋紵**
A: 鍙互锛屼絾閫熷害浼氬緢鎱€傚缓璁娇鐢?CPU 鎺ㄧ悊锛岃缁冨缓璁娇鐢?GPU銆?
**Q: 濡備綍浣跨敤鍏朵粬鍩哄骇妯″瀷锛?*
A: 淇敼 `config.json` 涓殑 `model.name`锛屼緥濡備娇鐢ㄦ洿澶х殑妯″瀷锛歚Qwen/Qwen2.5-1.5B-Instruct`銆?
**Q: 濡備綍娣诲姞鑷畾涔夋暟鎹紵**
A: 灏嗘暟鎹斁鍦?`data/` 鐩綍涓嬶紝鏍煎紡涓?JSONL锛屾瘡琛屽寘鍚?`{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`銆?
**Q: 璁粌鏁堟灉涓嶅ソ鎬庝箞鍔烇紵**
A: 鍙互灏濊瘯锛氬鍔犺缁冭疆鏁般€佹墿澶ф暟鎹泦銆佽皟鏁?LoRA rank銆佷娇鐢ㄦ洿澶х殑鍩哄骇妯″瀷銆?
## 璁稿彲璇?
鏈」鐩粎渚涘涔犲拰鐮旂┒浣跨敤銆
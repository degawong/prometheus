<!--
 * @Author: your name
 * @Date: 2021-09-07 16:10:20
 * @LastEditTime: 2021-09-08 17:02:29
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \prometheus\data\color_transfer\README.md
-->

`pip install color_transfer`

```python
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)
transfered_image = color_transfer(source, target)
transfered_image = color_transfer(source, target, clip='t', preserve_paper=False)
```
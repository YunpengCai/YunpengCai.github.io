---
layout:     post   				    # 使用的布局（不需要改）
title:      Thoughts in Leetcode				# 标题 
subtitle:    #副标题
date:       2018-12-10 				# 时间
author:     Yunpeng						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - python3
---
## 
>
2. Add Two Numbers (Medium)
Dealing with non-existing values:
if node.val not exist but want to assign it to 0
instead of using
if node: v1=node.val
if not node: v1=0
Try assign v1 to 0 in the beginning and only change its value when node exist to save space
v1  = 0
if node: v1 = node.val

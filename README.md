## DAB-DETR-demo





### Input

---

此处作者定义了NestedTensor类，假设输入batchsize=2的图片。

```python
im0 = torch.rand(3,200,200)
im1 = torch.rand(3,200,250)
x = nested_tensor_from_tensor_list([im0, im1])
```

​		NestedTensor 指的是把 {tensor, mask} 打包在一起，tensor就是图片的值，那么mask是什么呢？

​		当一个batch中的图片大小不一样的时候，我们要把它们处理的整齐，简单说就是把图片都padding成最大的尺寸，padding的方式就是补零，那么batch中的每一张图都有一个mask矩阵，所以mask大小为（2, 200, 250）, tensor大小为（2, 3, 200, 250）。



### Backbone

---

Backbone采用ResNet，输出（2, 1024, 24, 32），其次输出mask（2, 24, 32），mask采用F.interpolate()得到。



### position Embedding
---



**position embedding和position encoding的区别**

（1）position Embedding.

​		position embedding 在 Convolutional Sequence to Sequence Learning 定义。大意就是token序列$(x_0, x_1, x_2, ...)$经过 embedding 矩阵变成$(w_0, w_1, w_2, ...)$的词向量序列，同时绝对位置序列$(0, 1, 2, ...)$也经过另一个 embedding 矩阵变为$(p_0, p_1, p_2, ...)$。

​		最终的 embedding 序列就是$e = (x_0 + p_0, x_1 + p_1, x_2 + p_2, ...)$。position embeding 矩阵靠训练学习获得。

（2）position encoding.

​		position encoding 是 Attention is all you need 中使用的方法。Position Encoding（PE）计算公式如下：
$$
PE_{(pos, 2i)} = sin(\frac{pos}{10000^{2i / d_{model}}}) \\
PE_{(pos, 2i+1)} = cos(\frac{pos}{10000^{2i / d_{model}}})
$$


### Transformer

---

经过上面的Position Encoding之后，得到：src(2, 256, 24, 32)，mask(2, 24, 32)，pos(1, 2, 256, 24, 32)。

```python
hs = transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
```

hs是传入Transformer的输入，其中input_proj是一个卷积层，卷积核为1*1，就是压缩通道，将2048压缩到256，所以传入transformer的维度是压缩后的[2, 256, 24, 32]。self.query_embed.weight在decoder的时候用的到，后面再讲。



**TransformerEncoder**

​		此时可以知道Encoder部分由6个TransformerEncodeLayer组成，每个TransformerEncodeLayer由1个self_attention，2个ffn，2个norm。最后经过Encoder输出memory的size依然为(768, 2, 256)。



**TransformerDecoder**

先解释上面的self.query_embed.weight。

```python
# num_queries = 100 hidden_dim = 256
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

意思就是初始化一个(100, 256)的框向量（box-embeding）。



### Loss

---

输出100个预测框，假设其中有8个target，分别为t1(dog)，t2(cat)，t3(people)，... t8(car)。



（1）Loss Cost.

匹配部分，首先计算二分匹配图的cost-matrix，由3部分组成：

<img src="https://pic1.zhimg.com/80/v2-b0716087d6f11a99ea8e344a59c0c9d8_720w.jpg" style="zoom:67%;" />

这个图的意思没表达清楚。图中的$c_1, c_2, ..., c_{92}$实际上是coco数据集中的类别个数（91类），每个bbox输出的shape为(100, 92)，92是加上了noobject这个类别，输出的每个位置上是一个值，表示这一类别的概率。然后再与8个target进行计算。



1、cost_class，100个预测框取对应target的类别分数。

2、cost_bbox(100,8)，100个预测框和8个目标框间的L1距离。

3、cost_giou(100,8)，100个预测框和8个目标框间的giou值。
$$
costmatrix = \lambda_1 \ cost_{class} + \lambda_2 \ cost_{bbox} + \lambda_3 \ cost_{giou}
$$
再通过linear_sum_assignment计算最优匹配：

<img src="https://pic4.zhimg.com/80/v2-0c28104c89612d3ac3414bcdbb1c7c3f_720w.jpg" style="zoom:67%;" />

（2）Loss Function.

1、loss_labels

经过二分匹配，100个预测框对应的label为下图，然后计算100个预测框的CE。

![](https://pic2.zhimg.com/80/v2-be91efd98111daa02cbb1a0dfc4afcb5_720w.jpg)

2、loss_cardinality

计算每次100个预测框算出来的类别中是物体的个数与target中物体数（8个）计算L1-loss，该部分不参与反向传播，例如一个预测出100个框中有99个物体，那么该部分损失为99-8=91。

3、loss_boxes

计算每次100个预测中匹配到有物体的框（4，26，...，89）与对应的目标框（8个）计算L1-loss + giou-loss。





### References

---

DETR mask https://zhuanlan.zhihu.com/p/345985277

PositionEmbeddingSine https://zhuanlan.zhihu.com/p/361253913

CrossViT: Cross Attention https://zhuanlan.zhihu.com/p/361345433

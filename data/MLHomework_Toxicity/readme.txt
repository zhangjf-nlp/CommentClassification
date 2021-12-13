低分辨率视频行为识别 数据集


建议先查看下Civil Comments，数据是在Civil Comments上选取的


文件结构：
    |-- train.csv
    |-- test.csv
    |-- train_extra.csv
    |-- evaluate.py		:评估代码
    |-- submission.zip  :提交示例

    train.csv:
    训练数据，格式如下：
    {
        "comment_text":       语句文本(string)
        "target":             恶意程度(float)，通过多人投票平均得到。
                              target>0.5 被视为恶意(toxic)
    }

    test.csv:
    与train.csv相同，但是只包含"comments"，你需要预测出"target"。

    train_extra.csv:
    train的一些补充信息，这些信息你无法在test.csv中找到，但是他们可能会帮助你
    更有效的训练模型，并且会帮助你更好的理解这个问题。

    {
        # 以下几个为一些细分的类别，这些类别是为了之后的研究
        "severe_toxicity":
        "obscene":
        "threat":
        "insult":
        "identity_attack":
        "sexual_explicit":

        # 其他的一些为额外的标签，也是通过多人投票得到，是浮点数
        "male":               是否包含男性词汇
        "female":             是否包含女性词汇
        "bisexual":           是否包含双性恋词汇
        ...
        # 例如：Why would you assume that the nurses in this story were women?
        # 这句话最后得到的标签为：female: 0.8 (all others 0.0)
        # 例如：Continue to stand strong LGBT community. Yes, indeed, you'll overcome and you have.
        # 这句话最后得到的标签为：homosexual_gay_or_lesbian: 0.8, bisexual: 0.6, transgender: 0.3 (all others 0.0)

        # 还有一些元数据：
        "toxicity_annotator_count":     判断为恶意评论的人数
        "identity_annotator_count":     标注人数

        "publication_id":               在Civil Comments上的出版id
        "rating":                       Civil Comments上用户最终是否毙掉了这个评论
    }

    submission.txt
    提交样例

提交格式：
    id toxicity
    其中，id为test.json中的id，toxicity为浮点型数字，表示恶意程度。

提交示例：
	239653 0.300000
	239715 0.000000
	239792 0.000000
	239846 0.200000


    详见submission.zip

评估方式：
    ROC-AUC
    Bias AUCs



def find_hard_samples(classifier, target_dataloader, threshold=0.8):
    classifier.eval()
    hard_indices = []
    
    with torch.no_grad():
        for idx, (images, _) in enumerate(target_dataloader):
            logits = classifier(images)
            probs = torch.softmax(logits, dim=1)
            
            # 计算预测熵 (Entropy)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            # 找到模型“最迷茫”的样本 (熵值大的)
            if entropy > threshold:
                hard_indices.append(idx)
                
    return hard_indices # 这些就是自动发现的“黄金标注样本”
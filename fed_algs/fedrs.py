

def _local_training(self, party_id):
    net = self.party2nets[party_id]
    net.train()
    net.cuda()

    train_dataloader = self.party2loaders[party_id]
    test_dataloader = self.test_dl

    # compute restrict strength for each class
    n_class = net.classifier.out_features
    ds = train_dataloader.dataset
    uniq_val = np.unique(ds.target)
    class2data = self.appr_args.restricted_strength * torch.ones(n_class)
    for c in uniq_val:
        class2data[c] = 1.0
    class2data = class2data.unsqueeze(dim=0).cuda()

    self.logger.info('Training network %s' % str(party_id))
    self.logger.info('n_training: %d' % len(train_dataloader))
    self.logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader)
    test_acc, _ = compute_accuracy(net, test_dataloader)

    self.logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    self.logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=self.args.lr, momentum=self.args.rho, weight_decay=self.args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(self.args.epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            target = target.long()
            optimizer.zero_grad()
            out = net(x)

            # apply restricted softmax
            out *= class2data

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        self.logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _ = compute_accuracy(net, train_dataloader)
    test_acc, _ = compute_accuracy(net, test_dataloader)
    self.logger.info('>> Training accuracy: %f' % train_acc)
    self.logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
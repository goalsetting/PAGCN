

import torch
import torch.nn.functional as F
from tqdm import tqdm


from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = x.relu_()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.convs[1](x, edge_index))
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


# model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#
# def train(epoch):
#     model.train()
#
#     pbar = tqdm(total=int(len(train_loader.dataset)))
#     pbar.set_description(f'Epoch {epoch:02d}')
#
#     total_loss = total_correct = total_examples = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         y = batch.y[:batch.batch_size]
#         y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
#         loss = F.cross_entropy(y_hat, y)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += float(loss) * batch.batch_size
#         total_correct += int((y_hat.argmax(dim=-1) == y).sum())
#         total_examples += batch.batch_size
#         pbar.update(batch.batch_size)
#     pbar.close()
#
#     return total_loss / total_examples, total_correct / total_examples
#
#
# @torch.no_grad()
# def test():
#     model.eval()
#     y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
#     y = data.y.to(y_hat.device)
#
#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
#     return accs
#
#
# times = []
# for epoch in range(1, 11):
#     start = time.time()
#     loss, acc = train(epoch)
#     print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
#     train_acc, val_acc, test_acc = test()
#     print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#           f'Test: {test_acc:.4f}')
#     times.append(time.time() - start)
# print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
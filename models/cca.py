import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0")

def ca_weight(proj_query, proj_key):
    [b, c, h, w] = proj_query.shape
    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    energy_H = torch.bmm(proj_query_H, proj_key_H).view(b, w, h, h).permute(0, 2, 1, 3)
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(b, h, w, w)
    concate = torch.cat([energy_H, energy_W], 3)

    return concate

def ca_map(attention, proj_value):
    [b, c, h, w] = proj_value.shape
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    att_H = attention[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
    att_W = attention[:, :, :, h:h + w].contiguous().view(b * h, w, w)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
    out = out_H + out_W

    return out

class CrissCrossAttention(nn.Module):
    def __init__(self,in_dim):
        super(CrissCrossAttention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma*out + x

        return out

# if __name__ == '__main__':
#     data = torch.randn(4, 8, 512, 512).to(device)
#     cca = CrissCrossAttention(8).to(device)
#     out = cca(data)
#     print(out.shape)
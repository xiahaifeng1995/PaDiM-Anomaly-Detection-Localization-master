## choose Efficient model
from efficientnet_pytorch import EfficientNet

class EfficientNetModified(EfficientNet):
    '''
    The function of the existing model(original) will extract only the last layer feature. 
    This time, we're going to pull out the features that we want

    ref) https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    '''
    def extract_features(self, inputs, block_num):
        """ Returns list of the feature at each level of the EfficientNet """

        feat_list = []
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        
        iter = 1
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if iter in block_num:
                feat_list.append(x)
            iter += 1

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        # feat_list.append(F.adaptive_avg_pool2d(x, 1))
        # feat_list.append(x)

        return feat_list

    def extract_entire_features(self, inputs):
        """ Returns list of the feature at each level of the EfficientNet """

        feat_list = []
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # feat_list.append(F.adaptive_avg_pool2d(x, 1))
        feat_list.append(x)
        
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            feat_list.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        # feat_list.append(F.adaptive_avg_pool2d(x, 1))
        feat_list.append(x)

        return feat_list
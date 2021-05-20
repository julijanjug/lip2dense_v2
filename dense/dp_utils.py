
def RoIFeatureTransform(
        self,
        blobs_in,
        blob_out,
        blob_rois='rois',
        method='RoIPoolF',
        resolution=7,
        spatial_scale=1. / 16.,
        sampling_ratio=0
    ):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.
        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        # Single feature level
        # sampling_ratio is ignored for RoIPoolF
        # >> mislim da se tu klice roipoolf iz caffe
        xform_out = self.net.__getattr__(method)(
            [blobs_in, blob_rois], [blob_out],
            pooled_w=resolution,
            pooled_h=resolution,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio
        )

        # Only return the first blob (the transformed features)
        return xform_out[0] if isinstance(xform_out, tuple) else xform_out
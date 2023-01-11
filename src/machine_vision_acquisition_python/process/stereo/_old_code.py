# def disparity_to_pointcloud(
#     self,
#     disparity: cv2.Mat,
#     left_remapped: cv2.Mat,
#     min_disp: Optional[int] = None,
#     max_disp: Optional[int] = None,
# ) -> "PyntCloud":
#     """Convert raw disparity output to coloured pointcloud"""
#     log.warning(f"This code is experimental at best!")
#     if min_disp is not None and max_disp:
#         mask = np.ma.masked_inside(disparity, min_disp, max_disp)
#         disparity = mask.data
#     xyz = cv2.reprojectImageTo3D(disparity, self.params.Q, True)
#     points3D = np.reshape(xyz, (self.image_size[0] * self.image_size[1], 3))
#     colours = np.reshape(
#         left_remapped, (self.image_size[0] * self.image_size[1], 3)
#     )

#     data = np.concatenate(
#         [points3D, colours], axis=1
#     )  # Combines xyz and BGR (in that order)

#     # Clip outputs to Z values between 0.2 and 2.0
#     idx = np.logical_and(data[:, 2] < 2.0, data[:, 2] > 0.2)
#     data = data[idx]  # Only keep indicies that matched logical_and
#     # PyntCloud epxects a Pandas DF. Explicitly name columns
#     data_pd = pd.DataFrame.from_records(
#         data, columns=["x", "y", "z", "blue", "green", "red"]
#     )
#     # the merging will have converted the colour channels to floats. Revert them to uchar
#     data_pd = data_pd.astype(
#         {
#             "x": np.float32,
#             "y": np.float32,
#             "z": np.float32,
#             "blue": np.uint8,
#             "green": np.uint8,
#             "red": np.uint8,
#         }
#     )
#     cloud = PyntCloud(data_pd)
#     return cloud

class BraTSSliceDataset(Dataset):
    def __init__(self, root_dir, modality_suffix="_flair.nii.gz",
                 image_size=128, context_slices=3):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.modality_suffix = modality_suffix
        self.context_slices = context_slices
        self.radius = context_slices // 2

        self.volume_paths = sorted(self.root_dir.rglob(f"*{self.modality_suffix}"))
        self.slice_tuples = []

        for p in self.volume_paths:
            vol = nib.load(str(p)).get_fdata()
            D = vol.shape[2]

            # central 80%, but shrink by radius so neighbours exist
            z_start = int(0.1 * D) + self.radius
            z_end   = int(0.9 * D) - self.radius

            for z in range(z_start, z_end):
                self.slice_tuples.append((p, z))

    def __len__(self):
        return len(self.slice_tuples)

    def __getitem__(self, idx):
        path, z_center = self.slice_tuples[idx]
        vol = nib.load(str(path)).get_fdata().astype(np.float32)

        slices = []
        # collect neighbouring slices
        for dz in range(-self.radius, self.radius + 1):
            sl = vol[:, :, z_center + dz]

            # same normalization as before, per slice
            non_zero = sl[sl != 0]
            if non_zero.size > 0:
                mean = non_zero.mean()
                std = non_zero.std() if non_zero.std() > 0 else 1.0
                sl = (sl - mean) / std

            sl = np.clip(sl, -5, 5)
            sl = (sl + 5) / 10.0  # to [0, 1]

            slices.append(sl)

        # stack into (C, H, W)
        stack = np.stack(slices, axis=0)          # (C, H, W)
        stack = torch.from_numpy(stack).unsqueeze(0)  # (1, C, H, W)

        # resize to IMAGE_SIZE
        stack = F.interpolate(
            stack,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                               # (C, H, W)

        # map [0,1] → [-1,1]
        stack = 2.0 * stack - 1.0

        return stack  # (context_slices, H, W)
from django import forms


class ManualForm(forms.Form):
    """Counting with user-provided ROI/MOI (file upload or interactive canvas drawing)."""

    use_demo_files  = forms.BooleanField(label="Dùng bộ demo cam_5 có sẵn", required=False, initial=True)
    video_upload    = forms.FileField(label="Video nguồn", required=False)
    weights_upload  = forms.FileField(label="Model Weights", required=False)
    roi_upload      = forms.FileField(label="ROI File (.txt)", required=False)
    roi_json        = forms.CharField(widget=forms.HiddenInput, required=False)
    moi_upload      = forms.FileField(label="MOI Vectors (.txt)", required=False)
    moi_json        = forms.CharField(widget=forms.HiddenInput, required=False)
    movement_upload = forms.FileField(label="Movement Description (.txt)", required=False)

    video_clip_id = forms.IntegerField(label="Video Clip ID", initial=10)
    conf          = forms.FloatField(label="Conf", initial=0.25)
    class_conf    = forms.CharField(label="Ngưỡng theo lớp", initial="car=0.25,truck=0.75", required=False)
    imgsz         = forms.IntegerField(label="Imgsz", initial=1280)
    frame_stride  = forms.IntegerField(label="Frame Stride", initial=1, min_value=1)
    quick_preview = forms.BooleanField(label="Quick Preview", required=False, initial=True)
    max_frames    = forms.IntegerField(label="Số frame tối đa", initial=600, min_value=30, required=False)
    save_video    = forms.BooleanField(label="Lưu video visualize", required=False, initial=True)

    def clean(self):
        cleaned = super().clean()
        use_demo_files = cleaned.get("use_demo_files", False)
        video_upload   = cleaned.get("video_upload")
        weights_upload = cleaned.get("weights_upload")
        roi_upload     = cleaned.get("roi_upload")
        roi_json       = (cleaned.get("roi_json") or "").strip()
        quick_preview = cleaned.get("quick_preview", False)
        max_frames    = cleaned.get("max_frames")
        if not use_demo_files and not video_upload:
            self.add_error("video_upload", "Cần upload video hoặc bật bộ demo cam_5 có sẵn.")
        if not use_demo_files and not weights_upload:
            self.add_error("weights_upload", "Cần upload weights hoặc bật bộ demo cam_5 có sẵn.")
        if not use_demo_files and not roi_upload and not roi_json:
            self.add_error("roi_upload", "Cần upload file ROI hoặc vẽ ROI trên canvas.")
        if quick_preview and not max_frames:
            self.add_error("max_frames", "Cần nhập số frame tối đa khi bật Quick Preview.")
        return cleaned


class AutoForm(forms.Form):
    """Fully-automated counting: SAM/Grounded-SAM bootstraps ROI, trajectories generate MOI."""

    use_demo_files  = forms.BooleanField(label="Dùng bộ demo cam_5 có sẵn", required=False, initial=True)
    video_upload    = forms.FileField(label="Video nguồn", required=False)
    weights_upload  = forms.FileField(label="Model Weights", required=False)
    movement_upload = forms.FileField(label="Movement Description (.txt)", required=False)

    video_clip_id   = forms.IntegerField(label="Video Clip ID", initial=10)
    conf            = forms.FloatField(label="Conf", initial=0.25)
    class_conf      = forms.CharField(label="Ngưỡng theo lớp", initial="car=0.25,truck=0.75", required=False)
    imgsz           = forms.IntegerField(label="Imgsz", initial=1280)
    frame_stride    = forms.IntegerField(label="Frame Stride", initial=1, min_value=1)
    quick_preview   = forms.BooleanField(label="Quick Preview", required=False, initial=True)
    max_frames      = forms.IntegerField(label="Số frame tối đa", initial=600, min_value=30, required=False)
    save_video      = forms.BooleanField(label="Lưu video visualize", required=False, initial=True)
    use_grounding   = forms.BooleanField(label="Sử dụng Grounding-DINO", required=False, initial=True)
    grounding_model = forms.CharField(
        label="Grounding Model",
        initial="IDEA-Research/grounding-dino-base",
        required=False,
    )
    text_prompt = forms.CharField(
        label="Text Prompt",
        initial="road surface . traffic lane . intersection",
        required=False,
    )

    def clean(self):
        cleaned = super().clean()
        use_demo_files = cleaned.get("use_demo_files", False)
        video_upload   = cleaned.get("video_upload")
        weights_upload = cleaned.get("weights_upload")
        quick_preview  = cleaned.get("quick_preview", False)
        max_frames    = cleaned.get("max_frames")
        if not use_demo_files and not video_upload:
            self.add_error("video_upload", "Cần upload video hoặc bật bộ demo cam_5 có sẵn.")
        if not use_demo_files and not weights_upload:
            self.add_error("weights_upload", "Cần upload weights hoặc bật bộ demo cam_5 có sẵn.")
        if quick_preview and not max_frames:
            self.add_error("max_frames", "Cần nhập số frame tối đa khi bật Quick Preview.")
        return cleaned


# Backward-compat alias so the old index view / run_result still compile.
DemoForm = ManualForm

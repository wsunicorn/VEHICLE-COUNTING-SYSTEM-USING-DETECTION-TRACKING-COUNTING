from django import forms


class ManualForm(forms.Form):
    """Counting with user-provided ROI/MOI (file upload or interactive canvas drawing)."""

    video_upload    = forms.FileField(label="Video nguồn")
    weights_upload  = forms.FileField(label="Model Weights")
    roi_upload      = forms.FileField(label="ROI File (.txt)", required=False)
    roi_json        = forms.CharField(widget=forms.HiddenInput, required=False)
    moi_upload      = forms.FileField(label="MOI Vectors (.txt)", required=False)
    moi_json        = forms.CharField(widget=forms.HiddenInput, required=False)
    movement_upload = forms.FileField(label="Movement Description (.txt)", required=False)

    video_clip_id = forms.IntegerField(label="Video Clip ID", initial=10)
    conf          = forms.FloatField(label="Conf", initial=0.3)
    imgsz         = forms.IntegerField(label="Imgsz", initial=960)
    frame_stride  = forms.IntegerField(label="Frame Stride", initial=1, min_value=1)
    quick_preview = forms.BooleanField(label="Quick Preview", required=False, initial=True)
    max_frames    = forms.IntegerField(label="Số frame tối đa", initial=1200, min_value=30, required=False)
    save_video    = forms.BooleanField(label="Lưu video visualize", required=False, initial=True)

    def clean(self):
        cleaned = super().clean()
        roi_upload    = cleaned.get("roi_upload")
        roi_json      = (cleaned.get("roi_json") or "").strip()
        quick_preview = cleaned.get("quick_preview", False)
        max_frames    = cleaned.get("max_frames")
        if not roi_upload and not roi_json:
            self.add_error("roi_upload", "Cần upload file ROI hoặc vẽ ROI trên canvas bên dưới.")
        if quick_preview and not max_frames:
            self.add_error("max_frames", "Cần nhập số frame tối đa khi bật Quick Preview.")
        return cleaned


class AutoForm(forms.Form):
    """Fully-automated counting: Grounded-SAM/SAM bootstrap generates ROI/MOI."""

    video_upload    = forms.FileField(label="Video nguồn")
    weights_upload  = forms.FileField(label="Model Weights")
    movement_upload = forms.FileField(label="Movement Description (.txt)", required=False)

    video_clip_id   = forms.IntegerField(label="Video Clip ID", initial=10)
    conf            = forms.FloatField(label="Conf", initial=0.3)
    imgsz           = forms.IntegerField(label="Imgsz", initial=960)
    frame_stride    = forms.IntegerField(label="Frame Stride", initial=1, min_value=1)
    quick_preview   = forms.BooleanField(label="Quick Preview", required=False, initial=True)
    max_frames      = forms.IntegerField(label="Số frame tối đa", initial=1200, min_value=30, required=False)
    save_video      = forms.BooleanField(label="Lưu video visualize", required=False, initial=True)
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
        quick_preview = cleaned.get("quick_preview", False)
        max_frames    = cleaned.get("max_frames")
        if quick_preview and not max_frames:
            self.add_error("max_frames", "Cần nhập số frame tối đa khi bật Quick Preview.")
        return cleaned


# Backward-compat alias so the old index view / run_result still compile.
DemoForm = ManualForm

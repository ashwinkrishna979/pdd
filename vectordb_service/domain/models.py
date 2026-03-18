from pydantic import BaseModel


class FrameInfo(BaseModel):
    scene_id: int
    frame_no: int
    label: str
    filename: str
    filepath: str


class FrameSummary(BaseModel):
    visible_text: list[str]
    possible_actions: list[str]


class ProcessedFrame(BaseModel):
    frame_id: int
    compact_elements: list[str]
    summary: FrameSummary
    vector_metadata: dict


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    video_id: str | None = None


class FrameMatch(BaseModel):
    vector_id: str
    score: float
    metadata: dict
    image_url: str


class QueryResponse(BaseModel):
    query: str
    results: list[FrameMatch]


class LocateRequest(BaseModel):
    query: str
    top_k: int = 5
    video_id: str | None = None
    description: str | None = None


class LocateResponse(BaseModel):
    query: str
    selected_frame: FrameMatch
    selection_reason: str
    omniparser_elements_count: int
    located_element: str
    highlight_type: str
    region: list[float]
    region_reason: str
    annotated_image_url: str


class UploadResponse(BaseModel):
    video_id: str
    scenes_detected: int
    frames_extracted: int
    frames_indexed: int
    message: str

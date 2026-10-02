"""
Data loader v4.

* ``motion_prompt=True``  : v3 prompts, except that General-QA prompts now state
  the paper's rule (a tag only after an object is grounded at >= 2 timestamps)
  instead of asking for tags without any grounding.
* ``motion_prompt=False`` : no-tag control -- every mention of <motion/> is
  removed from the system prompts (pair with scripts/strip_motion_tags.py).
* ``piecewise_prompt=True`` : additionally documents the optional ``from``/``to``
  segment attributes and ``frame``/``ref`` attributes.
"""

from datasets import Dataset, DatasetDict, load_dataset

from data_loader_v3 import SYSTEM_PROMPT as SYSTEM_PROMPT_V3

GROUNDING = (
    "When you mention any related object, person, or specific visual element in the reasoning process, "
    "you must strictly follow the following format: "
    "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. "
)
MOTION_RULE = (
    "After the last grounded observation of an object that you have grounded at two or more timestamps, "
    "describe its motion trajectory using a self-closing motion tag with discrete attributes: "
    '`<motion obj="object_name" dir="DIRECTION" speed="SPEED" scale="SCALE"/>` '
    "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
    "SPEED is one of {stationary, slow, moderate, fast}, "
    "and SCALE is one of {approaching, stable, receding}. "
    "Do not emit a motion tag for an object grounded at only one timestamp. "
)
PIECEWISE_RULE = (
    "If the motion changes direction, emit one tag per segment with "
    '`from="t1" to="t2"` attributes (seconds). Optional attributes: `frame="scene"` when the motion is '
    'described after compensating for camera motion (default `frame="image"`), and `ref="other_object"` '
    "when the motion is relative to another grounded object. "
)
HEAD = (
    "A conversation between user and assistant. The user provides a video and asks a question, "
    "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
    "and then provide the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. "
)
HEAD_MCQ = HEAD.replace("asks a question", "asks a multiple-choice question")
FREE_TAIL = "The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
MCQ_TAIL = "Only output the correct option in the <answer> </answer> section."


def load_json_dataset(path):
    """Load a .json/.jsonl list of samples in memory.

    Dataset.from_json writes a cache copy first and aborts with "Not enough disk
    space" when the filesystem reports 0 free bytes (common on NFS with quotas).
    Columns are the union of keys over all rows (from_list would use only the
    first row's keys).
    """
    import json as _json
    from datasets import Dataset as _Dataset
    with open(path) as f:
        rows = [_json.loads(l) for l in f if l.strip()] if path.endswith(".jsonl") else _json.load(f)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return _Dataset.from_dict({k: [r.get(k) for r in rows] for k in keys})


def build_system_prompts(motion_prompt: bool = True, piecewise_prompt: bool = False) -> dict:
    motion = (MOTION_RULE + (PIECEWISE_RULE if piecewise_prompt else "")) if motion_prompt else ""
    prompts = dict(SYSTEM_PROMPT_V3)
    prompts["temporal-spatial free-form QA"] = (
        HEAD + "All reasoning must be grounded in visual evidence from the video. " + GROUNDING + motion + FREE_TAIL)
    prompts["General video QA Free-form"] = HEAD + GROUNDING + motion + FREE_TAIL
    prompts["General video QA MCQ"] = HEAD_MCQ + GROUNDING + motion + MCQ_TAIL
    return prompts


def get_data(script_args):
    prompts = build_system_prompts(getattr(script_args, "motion_prompt", True),
                                   getattr(script_args, "piecewise_prompt", False))

    def make_conversation(example):
        task = example.get("task")
        if task == "visual QA":
            content = [{"type": "image"}, {"type": "text", "text": example["question"]}]
        elif task in prompts:
            content = [{"type": "video"}, {"type": "text", "text": example["question"]}]
        else:
            raise ValueError(f"Unknown task: {task}")
        example["prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": prompts[task]}]},
            {"role": "user", "content": content},
        ]
        return example

    if script_args.dataset_name.endswith((".json", ".jsonl")):
        dataset = DatasetDict({"train": load_json_dataset(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(make_conversation)
    train = dataset["train"]
    dataset["train"] = train.select(range(len(train) - len(train) % 4))
    print(f"Dataset 'train' split size: {len(dataset['train'])}")
    return dataset

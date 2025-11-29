# llava

import sys, os, io, re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 在torch之前
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 在torch之前
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # 在torch之前
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # 在torch之前
# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"  # 在torch之前

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))

import re, torch
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
from tqdm import tqdm

from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos

from transformers import AutoTokenizer, AutoModelForCausalLM





prompt_flag = ''
addition_prompt = None
################################
p_1 = 'There may be Abuse,Arrest,Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting，Shoplifting, Stealing, Vandalism 14 types of events such as abuse. Please carefully analyze these descriptions and combine them with possible events based on their scenarios'
p_2 = 'Please carefully analyze the video description provided below in combination with the three aspects: scene environment, characters/objects, and actions/behaviors.'
p_3 = 'Please carefully analyze the video description provided below in combination with the three aspects of ' \
      'information in the provided prior knowledge base: scene environment, characters/objects, and actions/behaviors.'
p_4 = 'There may be Abuse,Arrest,Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, ' \
      'Shooting，Shoplifting, Stealing, Vandalism 13 types of events. Please carefully analyze the video description ' \
      'provided below in combination with the three aspects: 1.scene environment, 2.characters/objects, ' \
      '3.actions/behaviors. Video description:'

if prompt_flag == '':
    summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

    context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?"
    # context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"

    format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."

#########################################
else:
    raise NotImplementedError


# think = False
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos",
                        help="root path of the video dataset, but in this script, it is not used really")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-tmp/model_hub/DeepSeek-R1-Distill-Qwen-14B",
                        help="checkpoint directory of the LLM model")
    parser.add_argument("--video_clip_summary_json", type=str, 
                        default='../result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_kmeans_1_0.4/LLaVA-Video-7B-Qwen2_ucf_prior_q_coarse/maxf64_ucf_prior_q_Here is a .json',
                        help="video summary json path (VLM results)"
                        )

    parser.add_argument("--prompt_flag", type=str,  # default=None,
        default=prompt_flag,
        help='prompt flag to indicate which prompt to use'
          )
    parser.add_argument("--addition_prompt", type=str,  # default=None,
        default=addition_prompt, 
        help="additional prompt to be added to prompt_flag's prompt"
        )
    parser.add_argument("--think", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--summary_prompt", type=str, default=summary_prompt)
    parser.add_argument("--context_prompt", type=str, default=context_prompt)
    parser.add_argument("--format_prompt", type=str, default=format_prompt)

    parser.add_argument("--output_summary_json", type=str,
        default = None,
        help="output video summary json path (LAVAD's video clip summary captions)", 
        )
    
    parser.add_argument("--captions_json", type=str, 
                        help="LAVAD's frame captions",
        default=None, )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--resume",type=bool,
        # default=True,
        default=False,
    )
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--score_summary", default=True,
        help="If True (VADTree), score the temporal summaries. If False, generate the temporal summaries.", )

    args = parser.parse_args()

    assert args.score_summary==True, 'Currently only support scoring the temporal summaries!'
    args.output_summary_json = args.video_clip_summary_json

    vadtree_path = os.path.dirname(os.path.dirname(__file__))
    if 'UCF' in args.output_summary_json:
        args.annotationfile_path = f'{vadtree_path}/dataset_info/ucf_crime/annotations/anomaly_test.txt'
    elif 'XD' in args.output_summary_json:
        args.annotationfile_path = f'{vadtree_path}/dataset_info/xd_violence/annotations/anomaly_test.txt'
    elif 'MSAD' in args.output_summary_json:
        args.annotationfile_path=f'{vadtree_path}/dataset_info/msad/annotations/anomaly_test.txt'
    else:
        raise ValueError('Unknown annotationfile_path file format.')

    args.output_scores_json=f'{os.path.dirname(args.output_summary_json)}/' \
                            f'{os.path.basename(args.ckpt_dir)}{"_think" if args.think else ""}{args.prompt_flag}/{"_addition_prompt" if args.addition_prompt else ""}/{os.path.basename(args.output_summary_json)}'

    if args.score_summary:
        if not (args.context_prompt + args.format_prompt and args.output_scores_json):
            parser.error(
                "--context_prompt, --format_prompt, and --output_scores_json are required for scoring the temporal summaries.")
    else:
        raise NotImplementedError
        if not (args.captions_json and args.summary_prompt):
            parser.error("--captions_json and --summary_prompt are required for generating the temporal summaries.")

    return args

def serialize(obj):
    """递归序列化对象为JSON兼容类型"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple, set)):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, io.IOBase):  # 处理文件对象
        return {
            "filename": obj.name,
            "mode": obj.mode,
            "closed": obj.closed
        }
    elif hasattr(obj, "__dict__"):  # 处理自定义对象和Namespace
        return serialize(vars(obj))
    else:
        return str(obj)

class LLMAnomalyScorer:
    def __init__(self, video_root, annotationfile_path, batch_size, frame_interval, summary_prompt, context_prompt,
            format_prompt, output_scores_json, output_summary_json, captions_json, ckpt_dir, tokenizer_path,
            temperature, top_p, max_seq_len, max_gen_len, args, ):
        self.video_root = video_root
        self.annotationfile_path = annotationfile_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.summary_prompt = summary_prompt
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.output_scores_json = output_scores_json
        self.output_summary_json = output_summary_json
        self.captions_json = captions_json
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.args = args
        self.cur_vid_preface_content = [-1, -1]
        # 加载output_summary_json 文件，所有视频的summary在这一个json中
        with open(self.output_summary_json) as f:
            self.all_summaries_data = json.load(f)
            print('all_summaries_data config :', self.all_summaries_data['config'])

        # self.generator = Llama.build(
        #     ckpt_dir=self.ckpt_dir,
        #     tokenizer_path=self.tokenizer_path,
        #     max_seq_len=self.max_seq_len,
        #     max_batch_size=self.batch_size,
        # )

        self.model = AutoModelForCausalLM.from_pretrained(self.ckpt_dir,  # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16, device_map="auto", )
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side="left")

        print(f'{"-" * 50} {self.ckpt_dir} loaded {"-" * 50}')

    def _prepare_dialogs(self, captions, batch_clip_idxs, is_summary, prompt=None, his=None):
        if is_summary:
            if prompt == None:
                prompt = self.context_prompt + " " + self.format_prompt
            # batch_clip_caption = [f"{captions[str(idx)]}." for idx in batch_clip_idxs]
            batch_clip_caption = [f"{captions[idx]}" for idx in batch_clip_idxs]
        else:
            raise NotImplementedError
            prompt = self.summary_prompt
            batch_clip_caption = ["\n ".join([captions[str(idx)][str(clip_idx)] for clip_idx in captions[str(idx)]])
                for idx in batch_clip_idxs]
        dialogs = []
        for idx, clip_caption_ in zip(batch_clip_idxs, batch_clip_caption):
            if self.args.addition_prompt != None:
                assert '3context' not in args.prompt_flag
                if idx == 0:
                    self.cur_vid_preface_content[0] = 0
                    clip_caption = f'[{clip_caption_}]'
                    self.cur_vid_preface_content[1] = clip_caption_
                else:
                    self.cur_vid_preface_content[0] = idx
                    clip_caption = f'[{clip_caption_}] ' + self.args.addition_prompt + f'[{self.cur_vid_preface_content[1]}]'
                    self.cur_vid_preface_content[1] = self.cur_vid_preface_content[
                                                          1] + ' ' + clip_caption_  # # 加上当前段的描述存下来
            else:
                clip_caption = clip_caption_

            # llama_p = 'You are an AI assistant that helps me review harmful information in videos.'
            if '3context' in args.prompt_flag:
                assert self.args.addition_prompt == None
                # Description of the previous segment:
                # Description of the current segment:
                # Description of the next segment:
                previous = captions[idx-1] if idx !=0 else 'The current video segment is the beginning segment, so there is no previous segment content.'
                next =captions[idx+1] if idx != len(captions)-1 else 'The current video segment is the last segment, so there is no next segment content.'
                content = prompt + f'\nDescription of the previous segment:{previous}\nDescription of the previous segment:{clip_caption}\nDescription of the previous segment:{next}\n'
            else:
                content = prompt + clip_caption

            if his!=None:
                dialogs.append([
                    *his,
                    {"role": "user", "content": content}, ])
            else:
                dialogs.append([
                    # {"role": "system", "content": 'You are an AI assistant that helps me review harmful information in videos.'},
                    {"role": "user", "content": content}, ])
        return dialogs

    def _generate_temporal_summaries(self, video, video_captions):
        raise NotImplementedError
        temporal_summaries = {}

        for batch_start_frame in tqdm(range(0, video.num_frames, self.batch_size * self.frame_interval),
                desc=f"Processing {video}", unit="batch", ):
            batch_end_frame = min(batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames)
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)

            dialogs = self._prepare_dialogs(video_captions, batch_frame_idxs, is_summary=False)

            results = self.generator.chat_completion(dialogs, max_gen_len=self.max_gen_len,
                temperature=self.temperature, top_p=self.top_p, )

            for result, clip_frame_idx in zip(results, batch_frame_idxs):
                temporal_summaries[str(clip_frame_idx)] = result["generation"]["content"].split("\n")[-1]

        return temporal_summaries

    def _parse_score(self, response):
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        if match:
            score = float(match.group(1))
        else:
            score = -1
            print(f'\t <Warning! Response not follow the rules:> {response}')
        return score

    def _parse_sum(self, response):
        pattern = r"<([^<>]*)>"
        match = re.search(pattern, response)
        if match:
            sum_ = match.group(1)
        else:
            sum_ = ''
            print(f'\t <Warning! Response not follow the sum rules:> {response}')
        return sum_

    def _parse_dur(self, response):
        out  = response.split('</think>')[-1]
        # match = re.search(pattern, out)
        if 'first_half' in out:
            dur = 'first_half'
        elif 'second_half' in out:
            dur = 'second_half'
        elif 'entire' in out:
            dur = 'entire'
        else:
            print(f'\t <Warning! Response <dur> not follow the rules:> {response}')
            dur = 'entire'
        return dur

    def _interpolate_unmatched_scores(self, scores):
        valid_scores = [(idx, score) for idx, score in scores.items() if score != -1]
        video_scores = np.interp(list(scores.keys()), *zip(*valid_scores))

        return dict(zip(scores.keys(), video_scores))

    def _score_temporal_summaries(self, video, video_name):
        video_scores = {}
        vid_sum_dict = self.all_summaries_data['vid_captions'][f"{video_name}.mp4"]
        # vid_path = os.path.join(f'{video_name}.{vid_seg[0]}_{vid_seg[1]}.mp4')
        clip_edge_list = list(vid_sum_dict.keys())
        captions_list = list(vid_sum_dict.values())  # captions
        for batch_start_clip in tqdm(range(0, len(vid_sum_dict), self.batch_size),
                desc=f"{self.batch_size} batch processing {video}", unit="batch",
                disable=True):
            batch_end_clip = min(batch_start_clip + self.batch_size, len(vid_sum_dict))
            batch_clip_idxs = range(batch_start_clip, batch_end_clip)

            dialogs = self._prepare_dialogs(captions_list, batch_clip_idxs, is_summary=True)

            # results = self.generator.chat_completion(
            #     dialogs,
            #     max_gen_len=self.max_gen_len,
            #     temperature=self.temperature,
            #     top_p=self.top_p,
            #     video_path = video,
            #     clip_edge_list = clip_edge_list,
            #     batch_clip_idxs = batch_clip_idxs
            # )

            text_batch = self.tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True, )
            if not self.args.think:
                text_batch = [i + '\n</think>\n\n' for i in text_batch]
            model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

            generated_ids_batch = self.model.generate(**model_inputs_batch, max_new_tokens=self.max_gen_len,
                temperature=self.temperature,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id, do_sample=True,
                top_k=50, top_p=self.top_p, )
            generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
            results = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

            for result, frame_idx in zip(results, batch_clip_idxs):
                response = result
                if self.args.think:
                    if '</think>' not in response:
                        print('-----------</think>  not in response !!!')
                    out  = response.split('</think>')[-1]
                    score = self._parse_score(out)
                else:
                    if '</think>'  in response:
                        print(f'!!!Not think setting but </think>  in response !!!')
                    score = self._parse_score(response)

                if 'sum' in self.args.prompt_flag:
                    sum_ = self._parse_sum(out)
                    video_scores[clip_edge_list[frame_idx]] = [score, sum_, response]
                    continue

                if 'bin' in self.args.prompt_flag:
                    if 'dur' in self.args.prompt_flag and score==1:
                        dur = self._parse_dur(response)
                    else:
                        dur=None
                    video_scores[clip_edge_list[frame_idx]] = [score, dur, response]
                    continue
                video_scores[clip_edge_list[frame_idx]] = [score, response]

        # video_scores = self._interpolate_unmatched_scores(video_scores)
        # print(video_scores)
        return video_scores

    def _mutil_step_score(self, video, video_name):
        video_sums = {}
        video_scores = {}
        vid_sum_dict = self.all_summaries_data['vid_captions'][f"{video_name}.mp4"]
        # vid_path = os.path.join(f'{video_name}.{vid_seg[0]}_{vid_seg[1]}.mp4')
        clip_edge_list = list(vid_sum_dict.keys())
        captions_list = list(vid_sum_dict.values())  # captions
        for batch_start_clip in tqdm(range(0, len(vid_sum_dict), self.batch_size),
                desc=f"{self.batch_size} batch processing {video}", unit="batch",
                disable=True):
            batch_end_clip = min(batch_start_clip + self.batch_size, len(vid_sum_dict))
            batch_clip_idxs = range(batch_start_clip, batch_end_clip)


            #################第一轮
            sum_dialogs = self._prepare_dialogs(captions_list, batch_clip_idxs, is_summary=True, prompt=self.summary_prompt)
            sum_text_batch = self.tokenizer.apply_chat_template(sum_dialogs, tokenize=False, add_generation_prompt=True, )
            if not self.args.think:
                sum_text_batch = [i + '\n</think>\n\n' for i in sum_text_batch]
            sum_model_inputs_batch = self.tokenizer(sum_text_batch, return_tensors="pt", padding=True).to(self.model.device)
            sum_generated_ids_batch = self.model.generate(**sum_model_inputs_batch, max_new_tokens=self.max_gen_len,
                temperature=self.temperature,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id, do_sample=True,
                top_k=50, top_p=self.top_p, )
            sum_generated_ids_batch = sum_generated_ids_batch[:, sum_model_inputs_batch.input_ids.shape[1]:]
            sum_results = self.tokenizer.batch_decode(sum_generated_ids_batch, skip_special_tokens=True)

            for result, frame_idx in zip(sum_results, batch_clip_idxs):
                if self.args.think:
                    if '</think>' not in result:
                        print('-----------</think>  not in response !!!')
                    out  = result.split('</think>')[-1]
                else:
                    assert '</think>' not in result
                    out = result
                video_sums[clip_edge_list[frame_idx]] = [out,result]
            # for i in sum_dialogs:
            #     i.append({"role": "system", "content": result})
            #     i.append({"role": "user", "content": self.context_prompt + self.format_prompt})

            if self.args.prompt_flag == '_sum3':
                continue
            #################第二轮
            # dialogs = sum_dialogs
            dialogs = self._prepare_dialogs(captions_list, batch_clip_idxs, is_summary=True)

            text_batch = self.tokenizer.apply_chat_template(dialogs, tokenize=False, add_generation_prompt=True, )
            if not self.args.think:
                text_batch = [i + '\n</think>\n\n' for i in text_batch]
            model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

            generated_ids_batch = self.model.generate(**model_inputs_batch, max_new_tokens=self.max_gen_len,
                temperature=self.temperature,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id, do_sample=True,
                top_k=50, top_p=self.top_p, )
            generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
            results = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

            for result, frame_idx in zip(results, batch_clip_idxs):
                response = result
                if self.args.think:
                    if '</think>' not in response:
                        print('-----------</think>  not in response !!!')
                    out  = response.split('</think>')[-1]
                    score = self._parse_score(out)
                else:
                    assert '</think>' not in response
                    score = self._parse_score(response)

                if self.args.prompt_flag == '_sum':
                    sum_ = self._parse_sum(out)
                    video_scores[clip_edge_list[frame_idx]] = [score, sum_, response]
                    continue

                if 'bin' in self.args.prompt_flag:
                    if 'dur' in self.args.prompt_flag and score==1:
                        dur = self._parse_dur(response)
                    else:
                        dur=None
                    video_scores[clip_edge_list[frame_idx]] = [score, dur, response]
                    continue
                video_scores[clip_edge_list[frame_idx]] = [score, response]

        # video_scores = self._interpolate_unmatched_scores(video_scores)
        # print(video_scores)
        return video_sums, video_scores
    def process_video(self, video, score_summary):
        video_name = video

        if not score_summary:
            raise NotImplementedError
            # Generate temporal summaries
            video_caption_path = Path(self.captions_json) / f"{video_name}.json"
            with open(video_caption_path) as f:
                video_captions = json.load(f)

            output_path = Path(self.output_summary_json) / f"{video_name}.json"

            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporal_summaries = self._generate_temporal_summaries(video, video_captions)
                with open(output_path, "w") as f:
                    json.dump(temporal_summaries, f, indent=4)
        else:
            # Score temporal summaries
            if self.args.prompt_flag in ['_sum1', '_sum2', '_sum3']:
                _sum, clip_scores = self._mutil_step_score(video, video_name)
                return (_sum, clip_scores)
            # elif self.args.prompt_flag == '_sum2':
            #     _sum, clip_scores = self._sum2_score(video, video_name)
            #     return (_sum, clip_scores)
            else:
                clip_scores = self._score_temporal_summaries(video, video_name)
                return (clip_scores,)  # output_path = Path(self.output_scores_json) / f"{video_name}.json"  #
                # output_path.parent.mkdir(parents=True, exist_ok=True)  # with open(output_path, "w") as f:  #     json.dump(video_scores, f, indent=4)


def run(video_root, annotationfile_path, batch_size, frame_interval, summary_prompt, context_prompt, format_prompt,
        output_scores_json, output_summary_json, captions_json, ckpt_dir, tokenizer_path, temperature, top_p,
        max_seq_len, max_gen_len, resume, pathname, num_jobs, job_index, score_summary, args, ):

    if score_summary:
        os.makedirs(os.path.dirname(output_scores_json), exist_ok=True)
        config = serialize(vars(args))  
        print(f'Save to: {os.path.abspath(output_scores_json)}')
        if os.path.exists(output_scores_json) and not args.resume:
            raise NotImplementedError
    else:
        raise NotImplementedError

    video_list_ = [VideoRecord(x.strip().split(), video_root) for x in open(annotationfile_path)]
    video_list_ = list(np.array_split(video_list_, num_jobs)[job_index])
    video_list = [os.path.basename(i.path) for i in video_list_]


    #先处理异常视频
    if 'XD' in args.output_summary_json:
        video_list = sorted(
            video_list,
            key=lambda x:
            ("label_A" in x, x)  # 元组排序规则：(是否包含label_A, 键本身)
        )
    elif 'MSAD' in args.output_summary_json or 'UCF' in args.output_summary_json:
            video_list = sorted(
            video_list,
            key=lambda x:
            ("ormal" in x, x)  # 元组排序规则：(是否包含Normal/normal, 键本身)
        )
    else:
        raise NotImplementedError





    llm_anomaly_scorer = LLMAnomalyScorer(video_root=video_root, annotationfile_path=annotationfile_path,
        batch_size=batch_size, frame_interval=frame_interval,
        summary_prompt=summary_prompt, context_prompt=context_prompt,
        format_prompt=format_prompt, output_scores_json=output_scores_json,
        output_summary_json=output_summary_json, captions_json=captions_json,
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature,
        top_p=top_p, max_seq_len=max_seq_len, max_gen_len=max_gen_len,

        args=args, )
    # 按照文件标记部分推理，可用于多个程序并行推理
    pattern = r'_(\d+)_(\d+)\.json$'
    matched = re.search(pattern, os.path.basename(args.output_summary_json))
    if matched is not None:
        s, e = int(matched.group(1)), int(matched.group(2))
        video_list = video_list[s:e+1]

    video_list_ = list(llm_anomaly_scorer.all_summaries_data['vid_captions'].keys()) # 根据需要打分的summary罗列视频
    video_list = [i[:-4] for i in video_list_]
    # video_list = [i[:-4] for i in video_list_][:150]
    # print(f'150 vid:{video_list}')
    # print(video_list)
    # return
    json_result = {}
    json_result['config'] = config
    json_result['vid_score'] = {}
    
    # 其他探索性实验设置
    if args.prompt_flag in ['_sum1', '_sum2', '_sum3']:
        sum_json_result = {}
        sum_json_result['config'] = config
        sum_json_result['vid_sum'] = {}
        output_sum_json = os.path.join(os.path.dirname(output_scores_json),'sum.json')

    if args.resume:
        print('resume infer---')
        with open(output_scores_json, 'r', encoding='utf-8') as file:
            old_data = json.load(file)
        json_result['vid_score'] = old_data['vid_score']
    for idx, video in enumerate(video_list):
        if os.path.basename(video)+'.mp4' in json_result['vid_score'].keys():
            print(f'{idx:04d}', datetime.now().strftime("%H:%M:%S"), f'skip {video}')
            continue
        vid_out = llm_anomaly_scorer.process_video(video, score_summary)
        json_result['vid_score'][os.path.basename(video) + '.mp4'] = vid_out[-1]
        print(f'{idx:04d}', datetime.now().strftime("%H:%M:%S"), os.path.basename(video), (f'len(vid_score):{len(vid_out[-1])}'))
        with open(output_scores_json, 'w', encoding='utf-8') as json_file:
            json.dump(json_result, json_file, ensure_ascii=False, indent=4)

        # 其他探索性实验设置
        if args.prompt_flag in ['_sum1', '_sum2', '_sum3']:
            assert len(vid_out) == 2
            sum_json_result['vid_sum'][os.path.basename(video) + '.mp4'] = vid_out[0]
            with open(output_sum_json, 'w', encoding='utf-8') as json_file:
                json.dump(sum_json_result, json_file, ensure_ascii=False, indent=4)

    print(f'Save to: {os.path.abspath(output_scores_json)}')

if __name__ == "__main__":
    args = parse_args()
    run(video_root=args.video_root, annotationfile_path=args.annotationfile_path, batch_size=args.batch_size,
        frame_interval=args.frame_interval, summary_prompt=args.summary_prompt, context_prompt=args.context_prompt,
        format_prompt=args.format_prompt, output_scores_json=args.output_scores_json,
        output_summary_json=args.output_summary_json, captions_json=args.captions_json, ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path, temperature=args.temperature, top_p=args.top_p,
        max_seq_len=args.max_seq_len, max_gen_len=args.max_gen_len, resume=args.resume, pathname=args.pathname,
        num_jobs=args.num_jobs, job_index=args.job_index, score_summary=args.score_summary, args=args)

'''
{'0, 23': 0.2, '24, 47': 0.2, '48, 82': 0.2, '83, 179': 0.3, '180, 217': 0.2, '218, 245': 0.2, '246, 275': 0.2, '276, 305': 0.3, '306, 329': 0.3, '330, 356': 0.3, '357, 445': 0.3, '446, 477': 0.4, '478, 514': 0.4, '515, 536': 0.3, '537, 579': 0.3, '580, 586': 0.3, '587, 596': 0.4, '597, 670': 0.3, '671, 815': 0.4, '816, 927': 0.4, '928, 1037': 0.3, '1038, 1080': 0.3}
{'0, 23': 0.2, '24, 47': 0.2, '48, 82': 0.4, '83, 179': 0.3, '180, 217': 0.2, '218, 245': 0.2, '246, 275': 0.2, '276, 305': 0.3, '306, 329': 0.2, '330, 356': 0.3, '357, 445': 0.3, '446, 477': 0.4, '478, 514': 0.4, '515, 536': 0.3, '537, 579': 0.3, '580, 586': 0.4, '587, 596': 0.4, '597, 670': 0.3, '671, 815': 0.3, '816, 927': 0.4, '928, 1037': 0.3, '1038, 1080': 0.3}
{'0, 23': 0.2, '24, 47': 0.2, '48, 82': 0.4, '83, 179': 0.4, '180, 217': 0.2, '218, 245': 0.2, '246, 275': 0.2, '276, 305': 0.3, '306, 329': 0.2, '330, 356': 0.3, '357, 445': 0.3, '446, 477': 0.4, '478, 514': 0.4, '515, 536': 0.3, '537, 579': 0.3, '580, 586': 0.5, '587, 596': 0.4, '597, 670': 0.4, '671, 815': 0.3, '816, 927': 0.4, '928, 1037': 0.4, '1038, 1080': 0.3}

'''

'''

'''

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from chatcad.chat_bot import gpt_bot
from med_datasets.data_util.mimic_cxr_utils import *
from pipeline.eval.eval_metrics import eval_result_dir

task = None

validated = True

save_classifier_result = False

if __name__ == "__main__":
    if len(sys.argv) == 2:
        task = sys.argv[1]
        assert task in ["correction", "history", "template", "comparison"]
    output_dir = Path(f"chatcad_{task}_val_result") if task is not None else Path(f"chatcad_val_result")
    split = "test"
    data_dir = Path("/scratch/xinyangjiang/datasets/MIMIC-CXR/")
    if validated:
        assert split == "test"
        if task is not None:
            val_json = data_dir / f"{task}_instructions_val.json"
        else:
            val_json = data_dir / f"instructions_val.json"
        val_study_ids = list(json.load(open(val_json))["data"].keys())
        print(val_study_ids)
    # os.environ["ENDPOINT_URL"] = "https://gcrgpt4aoai9c.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview"
    # os.environ["ENDPOINT_URL"] = "https://gcrgpt4aoai9c.azurewebsites.net/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    os.environ["ENDPOINT_URL"] = "https://llmapp00.openai.azure.com/openai/deployments/gpt35/chat/completions?api-version=2023-07-01-preview"
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key is not None
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving inference results in {output_dir}")
    dicomid2label = create_id2label_dict(data_dir/"mimic-cxr-2.0.0-metadata.csv")
    studyid2split = create_id2split_dict(data_dir/"mimic-cxr-2.0.0-split.csv")
    studyid2path = create_id2path_dict(data_dir/"mimic-cxr-2.0.0-metadata.csv")
    metadata = pd.read_csv(data_dir/"mimic-cxr-2.0.0-metadata.csv")

    chatbot = gpt_bot(engine="gpt-3.5-turbo",api_key=api_key)
    chatbot.start()

    for patient_path in tqdm((data_dir/"files").glob("p*/p*"), total=len(list((data_dir/"files").glob("p*/p*")))):
        patient_id = patient_path.name
        for study_path in patient_path.glob("s*"):
            study_id = study_path.name
            if validated and (study_id not in val_study_ids):
                continue
            if save_classifier_result and f"{study_id}.txt" in os.listdir("classifier_result"):
                continue
            if studyid2split[study_id[1:]] != split and not save_classifier_result: continue
            out_file_path = output_dir / f"{study_id}.txt"
            if os.path.exists(out_file_path) and not save_classifier_result: continue
            if task == "comparison":
                current_report_path = studyid2path[study_id]
                previous_report_path = get_previous_report_path(Path(current_report_path), metadata)
                if previous_report_path is None: continue
                _, previous_findings, _ = parse_report(data_dir / previous_report_path)
                if len(previous_findings) == 0: continue
            elif task is not None:
                generated_path = data_dir / "files" / f"reports_{task}" / f"{study_id}.txt"
                if not os.path.exists(generated_path): continue
                generated_data = parse_generated(generated_path, task)
                if generated_data is None: continue
            image_path_list = [str(path)[len(str(data_dir)) + 1:] for path in list(study_path.glob("*.jpg"))]
            image_label_list = [dicomid2label[path.split('/')[-1][:-4]] for path in image_path_list]
            image_path_list = [image_path_list[i] for i in range(len(image_path_list)) if image_label_list[i] in ["PA", "AP"]]
            if len(image_path_list) == 0: continue
            report_path = data_dir / "files" / "reports" / patient_id[:3] / patient_id / f"{study_id}.txt"
            _, findings, _ = parse_report(report_path)
            gt = findings.lower().strip()
            if gt == "" and not save_classifier_result: continue
            report, _ = chatbot.report_en(data_dir / image_path_list[0], save_classifier_result=save_classifier_result)
            if save_classifier_result:
                continue
            report = report.strip()
            if task is None:
                f = open(out_file_path, 'w')
                f.write(f"PRED:\n{report.lower()}\n\nGT:\n{gt}\n")
                f.close()
            else:
                ref_record = f"user: User uploads a chest x-ray and asks for a diagnosis result\nassistant: {report}"
                if task == "template":
                    message = "Please fill the following report template base on the patient's chest x-ray image and the report you generated:\n"
                    template_str = generated_data["template"]
                    message += template_str
                    gt = generated_data["report"].lower().strip()
                elif task == "comparison":
                    message = "Please refine the report you generated referencing the patient's chest x-ray report from the last visit:\n"
                    current_report_path = studyid2path[study_id]
                    previous_report_path = get_previous_report_path(Path(current_report_path), metadata)
                    message += previous_findings
                elif task == "correction":
                    message = "Given a medical report with mistakes:\n{wrong_report}\nPlease revise this report based on the chest x-ray radiographs and these instructions:\n{instructions}"
                    incorrect_str = generated_data["incorrect_report"]
                    fix_str = generated_data["instruction"]
                    message = message.format(wrong_report=incorrect_str, instructions=fix_str)
                elif task == "history":
                    message = "The patient has the following other medical conditions and history:\n{history}\nPlease refine the patient's report you generated base on both the chest x-ray image and these additional information"
                    history_str = generated_data["history"]
                    message = message.format(history=history_str)
                pred = chatbot.chat_en(message=message, ref_record=ref_record)
                pred = pred.lower().strip()
                findings_idx = pred.find("findings:")
                if findings_idx >= 0:
                    pred = pred[findings_idx:]
                f = open(out_file_path, 'w')
                f.write(f"PRED:\n{pred}\n\nGT:\n{gt}\n")
                f.close()
    eval_result_dir(output_dir)



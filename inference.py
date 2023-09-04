from chat_bot import gpt_bot
import nibabel as nib


os.environ["ENDPOINT_URL"] = "https://gcrgpt4aoai9c.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview"
api_key = os.environ.get("OPENAI_API_KEY")


chatbot = gpt_bot(engine="gpt-3.5-turbo",api_key=api_key)
chatbot.start()


image_path = "imgs/examples/chest.jpg"

report = report_en(image_path)

print(report)
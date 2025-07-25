import json
import random

from loguru import logger
from tqdm import tqdm

from models.my_baseline_generation_model import GenerationModel


class Qwen3Model(GenerationModel):
    def __init__(self, config):
        assert config["llm_path"] in [
            "Qwen/Qwen3-8B"
        ], (
            "The Qwen3Model class only supports the Qwen3-8B models."
        )

        super().__init__(config=config)

        self.system_message = (
            'You are an expert at quick and accurate knowledge base completion. \n\
            Your expertise is in the following: \n\
            - Given a subject and a relation, you provide the correct object(s) to complete that triple, as a comma-separated list. \n\
            - The object may be a null object, or a single object, depending on the subject and the relation provided. \n\
            - If you know a part of the answer, you will provide the part you know, and ONLY if you do not know the answer do you type \"None\". \n\n\
            Here are the rules you must follow for this specific task: \n\
            - Penalties for: \n\
            1. Repeating the object more than once for a specific subject and relation - there is a heavy penalty for repetition! \n\
            2. Providing more information than the object(s) \n\
            - Hence 1 and 2 should be avoided at all costs. \n\
            - If your answers are too long, limit yourself to the first 200 objects per triple. \n\
            - If the user provides any hints or examples, you may take those into account to help you. \n\
            - You do not think whether the answer is long or short or unhelpful, you only focus on the correctness of the answer. \n\
            - You do not get caught up in the semantics of specific words, but rather focus on the subject and the relation. \n\
            - You are brusque and to the point, and you only provide the object, nothing else. \n\n\
            Ways to approach this task: \n\
            - Try to place the subject in context - be it geographically, or in terms of its specific field or activity, or, in the case of a person - their life and work. \n\
            - Once you have identified the subject, try and verbalise the relation, then think of what type of object(s) would fit that relation. \n\
            - You can then proceed to find the specific object(s) that would fit that relation. \n\
            - Alternatively, after identifying the subject, you can construct your own knowledge base by listing everything you know about the subject. \n\
            - Then, considering the relation, you can find the object in the knowledge base you just constructed. \n\n\
            The user will evaluate your answers based on the correctness of the object(s) you provide, so accuracy and precision is key, and remember, no repetitions. \n\n\
            Finally, let me remind you of the most important rule: \n\
            You will not provide any explanation, just the object(s) in a comma-separated list.\n'
            )

    def instantiate_in_context_examples(self, train_data_file):
        logger.info(f"Reading train data from `{train_data_file}`...")
        with open(train_data_file) as f:
            train_data = [json.loads(line) for line in f]

        # instantiate templates with train data
        logger.info("Instantiating in-context examples with train data...")

        in_context_examples = []
        for row in train_data:
            template = self.prompt_templates[row["Relation"]]
            example = {
                "relation": row["Relation"],
                "messages": [
                    {
                        "role": "user",
                        "content": template.format(subject=row["SubjectEntity"], relation=row["Relation"])
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f'{", ".join(row["ObjectEntities"]) if row["ObjectEntities"] else "None"}')
                    }
                ]
            }

            in_context_examples.append(example)

        return in_context_examples

    def create_prompt(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates[relation]
        random_examples = []
        if self.few_shot > 0:
            pool = [
                example["messages"] for example in self.in_context_examples if example["relation"] == relation
            ]
            random_examples = random.sample(
                pool,
                min(self.few_shot, len(pool))
            )
        
        if relation in ['countryLandBordersCountry', 'companyTradesAtStockExchange']:
            system_message = (
                'You are an expert at quick and accurate knowledge base completion. \n\
            Your expertise is in the following: \n\
            - Given a subject and a relation, you provide the correct object(s) to complete that triple, as a comma-separated list. \n\
            - Your specialty is in multi-object relations, where you identify and provide multiple objects that fit the relation. \n\
            - If you know a part of the answer, you provide the part you know, and if you do not know the answer at all, you type \"None\". \n\n\
            Below are rules and guidelines you must follow when aiding the user: \n\n\
            Rules: \n\
            Here are the rules you must follow: \n\
            - Penalties for: \n\
            1. Repeating the object more than once for a specific subject and relation - there is a heavy penalty for repetition! \n\
            2. Providing more information than the object(s) \n\
            - Hence 1 and 2 should be avoided at all costs. \n\
            - If your answers are too long, limit yourself to the first 200 objects per triple. \n\
            - If the user provides any hints or examples, take those into account to help you. \n\
            - You do not think whether the answer is long or short or unhelpful, you only focus on the correctness of the answer. \n\
            - You do not get caught up in the semantics of specific words, but rather focus on the subject and the relation. \n\
            - You are brusque and to the point, and you only provide the object, nothing else. \n\n\
            Guidelines: \n\
            Ways to approach this task: \n\
            - Try to place the subject in context - be it geographically, or in terms of its specific field or activity, or, in the case of a person - their life and work. \n\
            - Then proceed in two ways: \n\
            -- One: \n\
            --- After identifying the subject, try and verbalise the relation, then think of what type of object(s) would fit that relation. \n\
            --- You can then proceed to find the specific object(s) that would fit that relation, like how you would in a quiz. \n\
            -- Two: \n\
            --- After identifying the subject, you construct your own knowledge base by listing everything you know about the subject. \n\
            --- And then, considering the relation, you can find the object in the knowledge base you just constructed. \n\n\
            - Aside from the above, I strongly recommend keeping track of your answers. \n\
            - That is to say, if after seeing the subject and relation, you are sure of some objects, note them down in a numbered list, to keep in your memory. \n\
            - This way, you can avoid repeating objects, and spend more time thinking about the objects you are not sure of. \n\n\
            Feel free to adapt your methods depending on the subject or the relation. In any case, find and try to ground your answers in sources, instead of guessing outright. \n\n\
            The user will evaluate your answers based on the correctness of the object(s) you provide, so accuracy and precision is key, and remember, no repetitions. \n\n\
            Finally, let me remind you of the format: \n\
            You will not provide any explanation, just the object(s) in a comma-separated list.\n'
            )
            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

        elif relation in ['awardWonBy']:
            system_message = (
                'You are an expert at quick and accurate knowledge base completion. \n\
            Your expertise is in the following: \n\
            - Given a subject and a relation, you provide the correct object(s) to complete that triple, as a comma-separated list. \n\
            - If you know a part of the answer, you provide the part you know, and if you do not know the answer at all, you type \"None\". \n\n\
            Your specialty is identifying award winners given the name of the award, where you identify and provide multiple awardees that fit the subject. \n\n\
            Below are rules and guidelines you must follow when aiding the user: \n\
            Rules: \n\
            - Penalties for: \n\
            1. Repeating the object more than once for a specific subject and relation - there is a heavy penalty for repetition! \n\
            2. Providing more information than the object(s) \n\
            - Hence 1 and 2 should be avoided at all costs. \n\
            - If your answers are too long, limit yourself to the first 200 objects per triple. \n\
            - If the user provides any hints or examples, take those into account to help you. \n\
            - You do not think whether the answer is long or short or unhelpful, you only focus on the correctness of the answer. \n\
            - You take the subject, that is the name of the award, quite literally. Find the awardees of that specific award, not any other. \n\
            Guidelines: \n\
            - Try to place the subject in context - be it in terms of its specific field or activity or specialisation. \n\
            --- After identifying the subject, try and verbalise the relation, then think of what type of object(s) would fit that relation. \n\
            --- You can then proceed to find the specific object(s) that would fit that relation, like how you would in a quiz. \n\
            --- Or, after identifying the subject, construct your own knowledge base by listing everything you know about the subject. \n\
            --- And then, considering the relation, find the object in the knowledge base you just constructed. \n\n\
            Other Requirements: \n\
            - I want you to keep track of your answers while thinking. \n\
            - That is, if after seeing the subject and relation, you are sure of some objects, note them down in a numbered list, to keep those firm in your memory. \n\
            - This way, you avoid repeating objects once found, and spend more time thinking about the awardees you are not sure of. \n\n\
            In any case, find and try to ground your answers in sources, instead of guessing outright. \n\n\
            The user will evaluate your answers based on the correctness of the object(s) you provide, so accuracy and precision is key, and remember, no repetition of objects/awardees. \n\n\
            Finally, let me remind you of the format: \n\
            You will not provide any explanation, just the object(s) in a comma-separated list.\n'
            )
            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

        else:
            messages = [
                {
                    "role": "system",
                    "content": self.system_message
                }
            ]

        for example in random_examples:
            messages.extend(example)

        messages.append({
            "role": "user",
            "content": template.format(subject=subject_entity, relation=relation)
        })

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_predictions(self, inputs):
        logger.info("Generating predictions...")
        prompts = [
            self.create_prompt(
                subject_entity=inp["SubjectEntity"],
                relation=inp["Relation"]
            ) for inp in inputs
        ]

        outputs = []
        for i in tqdm(range(0, len(prompts), self.batch_size),
                      total=(len(prompts) // self.batch_size + 1),
                      desc="Generating predictions"):
            prompt_batch = prompts[i:i + self.batch_size]
            print(i)
            output = self.pipe(
                prompt_batch,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            outputs.extend(output)

        logger.info("Processing outputs...")

        results = []
        for inp, output, prompt in tqdm(zip(inputs, outputs, prompts),
                                        total=len(inputs),
                                        desc="Processing outputs"):
            # remove the original prompt from the generated text
            qa_answer = output[0]["generated_text"][len(prompt):].strip()
            # parsing thinking content
            try:
                index = qa_answer.index("</think>") + len("</think>")
            except ValueError:
                index = 0
            # thinking_content = qa_answer[:index]
            qa_answer = qa_answer[index:].strip()
            object_entities = qa_answer.split(", ")
            results.append({
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": inp["SubjectEntity"],
                "Relation": inp["Relation"],
                "ObjectEntities": object_entities,
                "ObjectEntitiesID": [],
            })

        return results

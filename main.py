from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

import os
from dotenv import load_dotenv

load_dotenv()

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field(description="Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, maria, fátima.")

class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = "Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."

    def _run(self, input: str) -> str:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        format_instructions = parser.get_format_instructions()

        template = PromptTemplate(
            template="""Você deve analisar o input: {input}
                         e extrair o nome do estudante informado. 
                         A saída deve estar no formato JSON: {format_instructions}""",
            input_variables=['input'],
            partial_variables={"format_instructions": format_instructions}
        )

        chain = template | llm | parser

        try:
            result = chain.invoke({"input": input})
            return result
        except Exception as e:
            return f"Erro ao processar o input: {str(e)}"

# pergunta = "Quais sao os dados da Ana?"

# resposta = DadosDeEstudante().run(pergunta)
# print(resposta)

dados_de_estudante = DadosDeEstudante()

tools = [
    Tool(
        name = dados_de_estudante.name,
        func = dados_de_estudante._run,
        description = dados_de_estudante.description 
    )
]

llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

prompt = hub.pull("hwchase17/openai-functions-agent")

agente = create_tool_calling_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agente, tools=tools, verbose=True)
agent_executor.invoke({"input": "Quais sao os dados da Ana?"})
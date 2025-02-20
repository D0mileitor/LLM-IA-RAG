import streamlit as st
from typing import Optional, Tuple
import re
from datetime import datetime
import graphviz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
from PIL import Image
import io
import uuid
import time

class FlowchartGenerator:
    def __init__(self):
        self.temp_dir = "temp_flowcharts"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def _get_unique_filename(self, prefix: str, extension: str) -> str:
        """Gera um nome de arquivo único usando timestamp e UUID"""
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return os.path.join(self.temp_dir, f"{prefix}_{unique_id}.{extension}")

    def _process_step_text(self, step: str) -> str:
        """
        Processa o texto do passo para garantir que números não fiquem sozinhos
        e mantém a formatação apropriada.
        """
        # Função auxiliar para verificar se uma string é um número
        def is_number(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        # Divide o texto em palavras
        words = step.split()
        processed_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            
            # Se a palavra atual é um número e está sozinha ou é a última palavra
            if is_number(current_word) and (i == len(words) - 1 or i == 0):
                # Se houver próxima palavra, junta com ela
                if i < len(words) - 1:
                    processed_words.append(f"{current_word} {words[i+1]}")
                    i += 2
                # Se for a última palavra, junta com a anterior se possível
                elif i > 0:
                    processed_words[-1] = f"{processed_words[-1]} {current_word}"
                    i += 1
                # Se for a única palavra, adiciona um rótulo
                else:
                    processed_words.append(f"Etapa {current_word}")
                    i += 1
            else:
                processed_words.append(current_word)
                i += 1

        return ' '.join(processed_words)

    def save_flowchart_as_pdf(self, steps: list[Tuple[str, str]], output_path: str) -> bool:
        """Salva o fluxograma como PDF"""
        temp_png = None
        try:
            # Cria um arquivo temporário para o PNG
            temp_png = self._get_unique_filename("temp", "png")
            
            # Gera o fluxograma em PNG primeiro
            if not self.create_graphviz_flowchart(steps, temp_png):
                return False
            
            # Adiciona pequeno delay para garantir que o arquivo está pronto
            time.sleep(0.5)
            
            # Cria PDF
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            
            # Adiciona título
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Fluxograma do Processo")
            
            # Tenta abrir a imagem
            png_path = f"{temp_png}.png"
            if not os.path.exists(png_path):
                st.error("Arquivo PNG temporário não foi criado")
                return False
                
            img = Image.open(png_path)
            img_width, img_height = img.size
            
            # Calcula escala para caber na página
            scale = min((width - 100) / img_width, (height - 100) / img_height)
            new_width = img_width * scale
            new_height = img_height * scale
            
            # Desenha a imagem e salva o PDF
            c.drawImage(png_path, 
                    50, height - new_height - 70,
                    width=new_width, height=new_height)
            
            c.save()
            img.close()  # Fecha explicitamente a imagem
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao salvar PDF: {e}")
            return False
        
        finally:
            # Limpa arquivos temporários com tentativas múltiplas
            if temp_png:
                png_path = f"{temp_png}.png"
                for _ in range(3):  # Tenta até 3 vezes
                    try:
                        if os.path.exists(png_path):
                            time.sleep(0.5)  # Pequeno delay entre tentativas
                            os.remove(png_path)
                        break
                    except Exception as e:
                        st.warning(f"Tentativa de remover arquivo temporário falhou: {e}")
                        time.sleep(0.5)  # Aguarda antes da próxima tentativa

    def _cleanup_temp_files(self):
        """Limpa arquivos temporários antigos com retry"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 3600:
                    for _ in range(3):  # Tenta até 3 vezes
                        try:
                            time.sleep(0.5)  # Pequeno delay entre tentativas
                            os.remove(filepath)
                            break
                        except Exception as e:
                            st.warning(f"Tentativa de remover arquivo temporário falhou: {e}")
                            time.sleep(0.5)  # Aguarda antes da próxima tentativa
        except Exception as e:
            st.warning(f"Erro ao limpar arquivos temporários: {e}")

    def create_graphviz_flowchart(self, steps: list[Tuple[str, str]], output_path: str) -> bool:
        """Cria um fluxograma usando Graphviz"""
        try:
            dot = graphviz.Digraph()
            dot.attr(rankdir='TB')
            
            # Configuração de estilo global
            dot.attr('node', 
                    shape='rect',
                    style='rounded,filled',
                    fontname='Arial',
                    fontsize='10',
                    height='1.0',
                    width='4.0',
                    fontcolor='white',
                    margin='0.3',
                    wrap='true')
            
            dot.attr('edge', 
                    color='#666666',
                    penwidth='1.5')
            
            # Adiciona nós e conexões
            total_steps = len(steps)
            for i, (step, _) in enumerate(steps):
                node_id = f'step{i}'
                
                # Processa o texto para caber melhor no nó
                step_text = step.upper()
                words = step_text.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) > 50:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                final_text = '\\n'.join(lines)
                node_color = '#4682B4'
                
                dot.node(node_id, 
                        final_text,
                        fillcolor=node_color,
                        color=node_color)
                
                if i > 0:
                    dot.edge(f'step{i-1}', node_id)
            
            dot.render(output_path, format='png', cleanup=True)
            time.sleep(0.5)
            return True
            
        except Exception as e:
            st.error(f"Erro ao criar fluxograma: {e}")
            return False
        
    def _extract_steps(self, text: str) -> list[Tuple[str, str]]:
        """Extrai passos do texto usando regex e quebras de linha"""
        steps = []
        
        # Divide o texto em sentenças usando pontuação comum
        sentences = re.split(r'[.!?]+', text)
        steps = [sent.strip() for sent in sentences if sent.strip()]
        
        # Se não encontrou sentenças, tenta dividir por quebras de linha
        if not steps:
            steps = [line.strip() for line in text.split('\n') if line.strip()]
            
        # Se ainda não encontrou, divide por ponto e vírgula
        if not steps:
            steps = [part.strip() for part in text.split(';') if part.strip()]
        
        # Remove duplicatas mantendo a ordem
        steps = list(dict.fromkeys(steps))
        
        # Formata as etapas aplicando o processamento de números
        formatted_steps = []
        for step in steps:
            # Simplifica mantendo a informação essencial
            simplified = ' '.join(step.split())  # Remove espaços extras
            if simplified:
                # Processa o texto para evitar números sozinhos
                processed = self._process_step_text(simplified)
                formatted_steps.append((processed, step))  # (versão processada, texto original)
        
        return formatted_steps

    def generate_response(self, question: str, response: str, output_format: str = 'text') -> tuple[str, Optional[str]]:
        """Gera a resposta no formato especificado (texto ou fluxograma)"""
        try:
            # Limpa arquivos temporários antigos
            self._cleanup_temp_files()
            
            if output_format == 'text':
                return response, None
            
            # Debug: Imprime o texto recebido
            st.write("Texto recebido:", response)
                
            steps = self._extract_steps(response)
            # Debug: Imprime os passos extraídos
            st.write("Passos extraídos:", steps)
            
            if not steps:
                return "Não foi possível extrair passos para gerar o fluxograma.", None
                
            # Gera nomes de arquivo únicos
            output_file = self._get_unique_filename("flowchart", 
                                                  "pdf" if output_format == 'pdf' else "png")
            
            success = False
            if output_format == 'pdf':
                success = self.save_flowchart_as_pdf(steps, output_file)
            else:  # png
                success = self.create_graphviz_flowchart(steps, output_file)
            
            if success:
                actual_file = f"{output_file}.png" if output_format == 'png' else output_file
                return response, actual_file
            else:
                return "Erro ao gerar o fluxograma. Por favor, tente novamente.", None
                
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")
            return "Ocorreu um erro ao gerar a resposta. Por favor, tente novamente.", None
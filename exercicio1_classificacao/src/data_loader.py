import pandas as pd
import os  

def load_data(file_path, label_col="Label", message_col="Message"):
    """
    Carrega um dataset (CSV ou TSV), identifica o separador pela extensão
    e retorna as colunas de texto e rótulo.
    
    Args:
        file_path (str): O caminho para o arquivo de dados.
        label_col (str): O nome da coluna de rótulos (Label).
        message_col (str): O nome da coluna de texto (Message).
    
    Returns:
        tuple: (texts, labels)
    """
    print(f"Carregando dados de: {file_path}")
    
    # Tenta adivinhar o separador pela extensão do arquivo
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.csv':
        sep = ','
        print(" Arquivo .csv detectado.")
    elif file_extension == '.tsv':
        sep = '\t'
        print(" Arquivo .tsv detectado.")
    else:
        print(" Extensão não reconhecida (ou ausente). Tentando '\\t' (tabulação) como padrão.")
        sep = '\t' # Padrão para o caso do SMSSpamCollection

    # Tenta carregar o arquivo
    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            header=None,  # Assumindo que os novos datasets também não têm cabeçalho
            names=[label_col, message_col],
            on_bad_lines='skip' # Ignora linhas mal formatadas
        )
    except Exception as e:
        print(f"    Erro crítico ao carregar o arquivo {file_path}: {e}")
        return [], [] 

    # Se o separador estava errado, o pandas geralmente lê tudo em uma só coluna.
    if df.shape[1] == 1 and sep == '\t':
        print(" Falha ao carregar como TSV (detectada 1 coluna). Tentando como CSV...")
        try:
            df = pd.read_csv(
                file_path,
                sep=',', # Tenta com vírgula
                header=None,
                names=[label_col, message_col],
                on_bad_lines='skip'
            )
        except Exception as e:
            print(f"    Erro crítico ao tentar como CSV: {e}")
            return [], []
            
    # Garante que as colunas esperadas existem
    if message_col not in df.columns or label_col not in df.columns:
        print(f"    Erro: Colunas '{message_col}' ou '{label_col}' não encontradas no arquivo.")
        return [], []

    # Limpeza básica e retorno dos dados
    # Garante que não há valores nulos que quebrarão o pré-processamento
    df = df.dropna(subset=[message_col, label_col])
    
    texts = df[message_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    
    print(f"Dados carregados: {len(texts)} amostras.")
    
    return texts, labels

# --- Bloco de Teste ---
if __name__ == "__main__":
    print("--- Testando o Módulo data_loader ---")
    
    # Teste 1: Seu arquivo original (sem extensão)
    texts_spam, labels_spam = load_data("../data/SMSSpamCollection")
    if texts_spam:
        print(f"  Amostra (Spam): {texts_spam[2]} | {labels_spam[2]}\n")
    
    # Teste 2: Um arquivo .csv (Crie um 'teste.csv' para verificar)
    # Exemplo:
    # ham,this is a csv message
    # spam,this is a csv spam
    
    # texts_csv, labels_csv = load_data("../data/teste.csv")
    # if texts_csv:
    #     print(f"  Amostra (CSV): {texts_csv[0]} | {labels_csv[0]}\n")
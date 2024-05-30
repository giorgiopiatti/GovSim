def get_pretty_name_llm(name):
    if "gpt-4-0125-preview" == name:
        return "GPT-4"
    elif "z-gpt-4-turbo-2024-04-09" == name:
        return "GPT-4"
    elif "gpt-4o-2024-05-13" == name:
        return "GPT-4o"
    elif "gpt-3.5-turbo-0125" == name:
        return "GPT-3.5"
    elif "gpt-3.5-turbo" == name:
        return "GPT-3.5"
    elif "claude-3-haiku-20240307" == name:
        return "Claude-3 Haiku"
    elif "claude-3-sonnet-20240229" == name:
        return "Claude-3 Sonnet"
    elif "claude-3-opus-20240229" == name:
        return "Claude-3 Opus"
    elif "mistral-large-2402" == name:
        return "Mistral Large"
    elif "mistral-medium-2312" == name:
        return "Mistral Medium"
    elif "TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GPTQ" == name:
        return "Mixtral-8x7B"
    elif "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" == name:
        return "Mixtral-8x7B"
    elif "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" == name:
        return "Mistral-7B"
    elif "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" == name:
        return "Mixtral-8x7B"
    elif "mistralai/Mixtral-8x7B-Instruct-v0.1" == name:
        return "Mixtral-8x7B"
    elif "mistralai/Mistral-7B-Instruct-v0.2" == name:
        return "Mistral-7B"
    elif "databricks/dbrx-instruct" == name:
        return "DBRX"
    elif "meta-llama/Llama-2-7b-chat-hf" == name:
        return "Llama-2-7B"
    elif "meta-llama/Llama-2-13b-chat-hf" == name:
        return "Llama-2-13B"
    elif "meta-llama/Llama-2-70b-chat-hf" == name:
        return "Llama-2-70B"
    elif "TheBloke/Llama-2-70B-Chat-GPTQ" == name:
        return "Llama-2-70B"
    elif "meta-llama/Meta-Llama-3-8B-Instruct" == name:
        return "Llama-3-8B"
    elif "meta-llama/Meta-Llama-3-70B-Instruct" == name:
        return "Llama-3-70B"
    elif "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ" == name:
        return "Llama-3-70B"
    elif "Qwen/Qwen1.5-72B-Chat" == name:
        return "Qwen-72B"
    elif "Qwen/Qwen1.5-0.5B-Chat" == name:
        return "Qwen-0.5B"
    elif "Qwen/Qwen1.5-1.8B-Chat" == name:
        return "Qwen-1.8B"
    elif "Qwen/Qwen1.5-4B-Chat" == name:
        return "Qwen-4B"
    elif "Qwen/Qwen1.5-MoE-A2.7B-Chat" == name:
        return "Qwen-MoE-A2.7B"
    elif "Qwen/Qwen1.5-7B-Chat" == name:
        return "Qwen-7B"
    elif "Qwen/Qwen1.5-14B-Chat" == name:
        return "Qwen-14B"
    elif "Qwen/Qwen1.5-32B-Chat" == name:
        return "Qwen-32B"
    elif "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4" == name:
        return "Qwen-32B"
    elif "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4" == name:
        return "Qwen-72B"
    elif "Qwen/Qwen1.5-110B-Chat-GPTQ-Int4" == name:
        return "Qwen-110B"
    elif "CohereForAI/c4ai-command-r-plus-4bit" == name:
        return "Command R+"
    else:
        return name


def get_model_size_version(name):
    if "gpt-4-0125-preview" == name:
        return "4"
    elif "z-gpt-4-turbo-2024-04-09" == name:
        return "4"
    elif "gpt-4o-2024-05-13" == name:
        return "4o"
    elif "gpt-3.5-turbo-0125" == name:
        return "3.5"
    elif "gpt-3.5-turbo" == name:
        return "3.5"
    elif "claude-3-haiku-20240307" == name:
        return "Haiku"
    elif "claude-3-sonnet-20240229" == name:
        return "Sonnet"
    elif "claude-3-opus-20240229" == name:
        return "Opus"
    elif "mistral-large-2402" == name:
        return "Large"
    elif "mistral-medium-2312" == name:
        return "Medium"
    elif "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4" == name:
        return "72B"
    elif "TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GPTQ" == name:
        return "8x7B"
    elif "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" == name:
        return "8x7B"
    elif "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" == name:
        return "7B"
    elif "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" == name:
        return "8x7B"
    elif "mistralai/Mixtral-8x7B-Instruct-v0.1" == name:
        return "8x7B"
    elif "mistralai/Mistral-7B-Instruct-v0.2" == name:
        return "7B"
    elif "databricks/dbrx-instruct" == name:
        return "16x8.25B"
    elif "meta-llama/Llama-2-7b-chat-hf" == name:
        return "7B"
    elif "meta-llama/Llama-2-13b-chat-hf" == name:
        return "13B"
    elif "meta-llama/Llama-2-70b-chat-hf" == name:
        return "70B"
    elif "TheBloke/Llama-2-70B-Chat-GPTQ" == name:
        return "70B"
    elif "meta-llama/Meta-Llama-3-8B-Instruct" == name:
        return "8B"
    elif "meta-llama/Meta-Llama-3-70B-Instruct" == name:
        return "70B"
    elif "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ" == name:
        return "70B"
    elif "Qwen/Qwen1.5-72B-Chat" == name:
        return "72B"
    elif "Qwen/Qwen1.5-0.5B-Chat" == name:
        return "0.5B"
    elif "Qwen/Qwen1.5-1.8B-Chat" == name:
        return "1.8B"
    elif "Qwen/Qwen1.5-4B-Chat" == name:
        return "4B"
    elif "Qwen/Qwen1.5-MoE-A2.7B-Chat" == name:
        return "MoE-A2.7B"
    elif "Qwen/Qwen1.5-7B-Chat" == name:
        return "7B"
    elif "Qwen/Qwen1.5-14B-Chat" == name:
        return "14B"
    elif "Qwen/Qwen1.5-32B-Chat" == name:
        return "32B"
    elif "Qwen/Qwen1.5-32B-Chat-GPTQ-Int4" == name:
        return "32B"
    elif "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4" == name:
        return "72B"
    elif "Qwen/Qwen1.5-110B-Chat-GPTQ-Int4" == name:
        return "110B"
    elif "CohereForAI/c4ai-command-r-plus-4bit" == name:
        return "102B"
    else:
        return name


def get_LLM_order():
    return [
        "Command R+",
        "DBRX",
        "Llama-2-7B",
        "Llama-2-13B",
        "Llama-2-70B",
        "Llama-3-8B",
        "Llama-3-70B",
        "Mistral-7B",
        "Mixtral-8x7B",
        "Mistral Medium",
        "Mistral Large",
        "Qwen-0.5B",
        "Qwen-1.8B",
        "Qwen-4B",
        "Qwen-7B",
        "Qwen-14B",
        "Qwen-32B",
        "Qwen-MoE-A2.7B",
        "Qwen-72B",
        "Qwen-110B",
        "Claude-3 Haiku",
        "Claude-3 Sonnet",
        "Claude-3 Opus",
        "GPT-3.5",
        "GPT-4",
        "GPT-4o",
    ]


def get_LLM_family(name):
    name = name.lower()
    name = name.replace("gptq", "")
    if "gpt" in name:
        return "GPT"
    elif "claude" in name:
        return "Claude 3"
    elif "mistral" in name:
        return "Mistral"
    elif "mixtral" in name:
        return "Mistral"
    elif "qwen" in name:
        return "Qwen"
    elif "llama-2" in name:
        return "Llama-2"
    elif "llama-3" in name:
        return "Llama-3"
    elif "dbrx" in name:
        return "DBRX"
    elif "command" in name:
        return "Command"
    else:
        return "Other"


def prepare_table(df_old, max_columns=[], min_columns=[], display_std=False):
    # Apply bold formatting to max values
    df = df_old.copy()

    max_value_no_api = {
        col: df_old.loc[df_old["llm.is_api"] == False, col].max() for col in max_columns
    }
    min_value_no_api = {
        col: df_old.loc[df_old["llm.is_api"] == False, col].min() for col in min_columns
    }

    max_value = {col: df_old[col].max() for col in max_columns}
    min_value = {col: df_old[col].min() for col in min_columns}

    def format_value_max(x, col):
        if not x["llm.is_api"]:
            if x[col] == max_value[col]:
                return "\\underline{{\\textbf{{{:.2f}}}}}".format(x[col])
            elif x[col] == max_value_no_api[col]:
                return "\\underline{{{:.2f}}}".format(x[col])
        else:
            if x[col] == max_value[col]:
                return "\\textbf{{{:.2f}}}".format(x[col])
        return "{:.2f}".format(x[col])

    def format_value_min(x, col):
        if not x["llm.is_api"]:
            if x[col] == min_value[col]:
                return "\\underline{{\\textbf{{{:.2f}}}}}".format(x[col])
            elif x[col] == min_value_no_api[col]:
                return "\\underline{{{:.2f}}}".format(x[col])
        else:
            if x[col] == min_value[col]:
                return "\\textbf{{{:.2f}}}".format(x[col])
        return "{:.2f}".format(x[col])

    for col in max_columns:
        df[col] = df_old.apply(
            lambda x: format_value_max(x, col),
            axis=1,
        )
    for col in min_columns:
        df[col] = df_old.apply(
            lambda x: format_value_min(x, col),
            axis=1,
        )

    if display_std:
        # now we need to append the std information
        for col in max_columns:
            if "mean" not in col:
                continue
            df[col] = df.apply(
                lambda x: f"{x[col]}\\tiny{{$\\pm${x[col.replace('mean', 'std')]:.2f}}}",
                axis=1,
            )
        for col in min_columns:
            if "mean" not in col:
                continue
            df[col] = df.apply(
                lambda x: f"{x[col]}\\tiny{{$\\pm${x[col.replace('mean', 'std')]:.2f}}}",
                axis=1,
            )
    return df


def prepare_table_delta(df_old, max_columns=[], min_columns=[], display_std=False):
    # Apply bold formatting to max and min values and check if values are increasing or decreasing
    df = df_old.copy()

    max_value_no_api = {
        col: df_old.loc[df_old["llm.is_api"] == False, col].max() for col in max_columns
    }
    min_value_no_api = {
        col: df_old.loc[df_old["llm.is_api"] == False, col].min() for col in min_columns
    }

    max_value = {col: df_old[col].max() for col in max_columns}
    min_value = {col: df_old[col].min() for col in min_columns}

    # Function to format values and check deltas for maximum highlighted columns
    def format_value_max(x, col):
        text = "\\delta{{{:.2f}}}".format(x[col])
        if not x["llm.is_api"]:
            if x[col] == max_value[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])
            elif x[col] == max_value_no_api[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])
        else:
            if x[col] == max_value[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])

        # Check for delta changes
        if x[col] > 0:
            text = text.replace("delta", "gooddelta")
        elif x[col] < 0:
            text = text.replace("delta", "baddelta")
        else:
            text = text.replace("delta", "nodelta")
        return text

    # Function to format values and check deltas for minimum highlighted columns
    def format_value_min(x, col):
        text = "\\delta{{{:.2f}}}".format(x[col])
        if not x["llm.is_api"]:
            if x[col] == min_value[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])
            elif x[col] == min_value_no_api[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])
        else:
            if x[col] == min_value[col]:
                text = "\\delta{{{:.2f}}}".format(x[col])

        # Check for delta changes
        if x[col] > 0:
            text = text.replace("delta", "gooddelta")
        elif x[col] < 0:
            text = text.replace("delta", "baddelta")
        else:
            text = text.replace("delta", "nodelta")
        return text

    for col in max_columns:
        df[col] = df.apply(lambda x: format_value_max(x, col), axis=1)
    for col in min_columns:
        df[col] = df.apply(lambda x: format_value_min(x, col), axis=1)

    return df

# Global constants
# special tokens
PAD_TOKEN = "#PAD#"
UNK_TOKEN = "#UNK#"
START_TOKEN = "#START#"
END_TOKEN = "#END#"
PAD_LENGTH = 20

SPECIAL_TOKEN_MAP = {  # Use negative index to avoid clashing with vocab
    PAD_TOKEN: -1,
    UNK_TOKEN: -2,
    START_TOKEN: -3,
    END_TOKEN: -4,
}

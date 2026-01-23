
import numpy as np
from openrec.postprocess.ctc_postprocess import BaseRecLabelDecode

def test_eos_behavior():
    # Mock class to mimic behavior
    class MockDecoder(BaseRecLabelDecode):
        def __init__(self):
            self.character = ["p", "s", "e", "u", "A", "B", "C"]
            self.get_ignored_tokens = lambda: [0, 1, 2] # pad, sos, eos

    decoder = MockDecoder()
    
    # 0=pad, 1=sos, 2=eos, 4=A, 5=B
    # Case 1: Normal [A, B, EOS, PAD, PAD]
    # Expected: "AB"
    pred1 = np.array([[4, 5, 2, 0, 0]])
    res1 = decoder.decode(pred1)[0][0]
    print(f"Case 1 Input: [A, B, EOS, PAD, PAD]")
    print(f"Case 1 Output: '{res1}'")
    
    # Case 2: Hallucination after EOS [A, B, EOS, C, A]
    # Expected: "AB" (if stopping at EOS), but likely "ABCA" if not
    pred2 = np.array([[4, 5, 2, 6, 4]])
    res2 = decoder.decode(pred2)[0][0]
    print(f"Case 2 Input: [A, B, EOS, C, A]")
    print(f"Case 2 Output: '{res2}'")
    
    if res2 == "AB":
        print("PASS: Decodes stops at EOS.")
    else:
        print("FAIL: Decoder ignored EOS position but continued decoding.")

if __name__ == "__main__":
    test_eos_behavior()

def primitive_prompt(x,g):
    return 'State: '+' '.join(map(str,x))+'\nPrevious instructions: -\nCurrent instruction: '+g+'\nReturn only the next state as eight space-separated octal digits.'


def residual_prompt(x,h,g):
    return 'Operator result: '+' '.join(map(str,x))+'\nPrevious instructions: '+(' '.join(h) if h else '-')+'\nCurrent instruction: '+g+'\nReturn only the context-corrected state as eight space-separated octal digits.'


def policy_prompt(w,h):
    return 'Program: '+' '.join(w)+'\nExecuted instructions: '+(' '.join(h) if h else '-')+'\nReturn only the next instruction, or STOP if all instructions have been executed.'

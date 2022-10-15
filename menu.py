import sys
import helper

def display(projList):
    menuItem = {}
    menuIdx = 0
    newline = 0
    
    print("------------------------------------------------------------------------------------")
    print("-                                Scale Project Menu                                -")
    print("------------------------------------------------------------------------------------")

    for menuIdx, name in enumerate(projList):
        if newline == 0:
            print("[{:0d}]: {:42s}".format(menuIdx, name.name), end=" ")
            menuItem[menuIdx] = name.name
            newline = 1
        else:
            print(f"[{menuIdx}]: {name.name}")
            menuItem[menuIdx] = name.name
            newline = 0 
            
        # menuIdx += 1
        
    print("------------------------------------------------------------------------------------")
    option = input("Enter your option (whole numbers only): ")
    
    if option.strip().isdigit():
        option = helper.s2d(option)
        
        if(option > menuIdx):
            print("Please enter a value between 0 -", menuIdx, "and try again")
            sys.exit(1)
        
    else:
        print("Invalid Input: Please enter whole numbers only")
    
    print()
    return(menuItem[option])
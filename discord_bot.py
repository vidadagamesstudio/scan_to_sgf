import discord
import requests
import shutil
import tsumego_detector
import os
import uuid
import private_info

root_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root_path, "tmp")
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)

nb_digit_filename = 4
problem_indicator_prefix = "P "
client = discord.Client()
@client.event
async def on_message(message):
    if(message.author.id != client.user.id):
        if message.channel.id == message.author.dm_channel.id: # dm only
            if(len(message.attachments)==1):
                attachement = message.attachments[0]
                if not attachement.content_type.startswith("image/") :
                    return
                
                just_name=attachement.filename.split(".")[0]
                url = attachement.url
                path_dir_user = os.path.join(output_dir, str(message.author.id))
                if(os.path.exists(path_dir_user)):
                    shutil.rmtree(path_dir_user)
                path_dir_problem = os.path.join(path_dir_user, just_name)
                os.makedirs(path_dir_problem)
                if url[0:26] == 'https://cdn.discordapp.com':
                    r= requests.get(url, stream = True)
                    
                    image_path = os.path.join(path_dir_user,attachement.filename)
                    image_debug_path = os.path.join(path_dir_user,just_name+"_debug."+attachement.filename.split(".")[1])
                    out_file =open(image_path,'wb')
                    shutil.copyfileobj(r.raw,out_file)
                    out_file.close()
                    sgf_data = tsumego_detector.image_scan(image_path,image_debug_path)
                    print(r)
                
                
                
                path_problem = os.path.join(path_dir_problem, just_name+".sgf")
                f = open(path_problem, "w", encoding="utf-8")
                if(sgf_data != None):
                    f.write(sgf_data)
                f.close()
                
                
                await message.channel.send(file=discord.File(image_debug_path))
                await message.channel.send(file=discord.File(path_problem))
                shutil.rmtree(path_dir_user)
            else:
                await message.channel.send("Bonjour je suis le HelperBot envoie moi des images et je te les convertirais en SGF !")



            


client.run(private_info.discord_bot_key)
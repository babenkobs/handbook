function removeEmptyLines(elem, onlyFirst = false) {
    if (elem.nodeName == '#text') { lines = elem.textContent.split('\n')
        while (lines[0] == '') {
            lines.splice(0,1)
        }
        while (lines[lines.length-1] == '') {
            lines.pop()
        }
        elem.textContent = lines.join('\n')
    }
}

contentMd = document.querySelector('.content-md')

if (contentMd)
    contentMd.childNodes.forEach(removeEmptyLines)

//Adding top header
topHeader = document.createElement('div')
topHeader.classList.add('topMenu')
document.body.prepend(topHeader)

//Adding light button
btnLight = document.createElement('a')
btnLight.innerHTML = '&#9728;'
btnLight.classList.add('lightModeBtn')
topHeader.appendChild(btnLight)

//Adding home button
if (location.pathname != '/handbook/') {
    btnHome = document.createElement('a')
    btnHome.innerHTML = '	&#10094;'
    btnHome.classList.add('homeBtn')
    btnHome.setAttribute('href', '/handbook/')
    topHeader.appendChild(btnHome)
}

//Adjust night mode button
function setCookie(cname, cvalue, exdays) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays*24*60*60*1000));
  let expires = "expires="+ d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for(let i = 0; i <ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

if (getCookie('Night') == 'False') {
    document.body.classList.remove('night')
}

document.querySelector('.lightModeBtn').onclick = function(){
    document.body.classList.toggle('night')
    if (getCookie('Night') == 'False') {
        setCookie('Night', 'True')
    } else {
        setCookie('Night', 'False')
    }
}

// Adding copy function for code

copyElems = document.querySelectorAll('.copy')

copyElems.forEach((elem)=>{
    elem.onclick = function(){
        console.log(this.innerHTML)

        const element = this;
        const storage = document.createElement('textarea');
        storage.value = element.innerHTML;
        element.appendChild(storage);

        // Copy the text in the fake `textarea` and remove the `textarea`
        storage.select();
        storage.setSelectionRange(0, 99999);
        document.execCommand('copy');
        element.removeChild(storage);
    }
})
